#include "sr_engine.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "dsps_dotprod.h"
#include "sr_simd_pie.h"
#include "sr_parallel.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "SR_ENGINE";

struct sr_engine {
  sr_model_t model;
  size_t max_frames;
  bool is_fp16;

  // Ping-pong buffers for [max_frames, 512]
  float *buf_512_A;
  float *buf_512_B;
  float *buf_512_C;
  float *buf_512_x2;
  float *buf_512_x3;
  float *buf_512_x4;

  // Res2Net branch buffers [max_frames, 64]
  float *buf_branch_in;
  float *buf_branch_out;

  // Multi-layer aggregation buffer [max_frames, 1536]
  float *buf_1536_cat;
  float *buf_1536_agg;

  // Attentive Statistics Pooling buffers
  float *buf_att_in; // [max_frames, 4608]
  float *buf_att_h1; // [max_frames, 128]
  float *buf_att_h2; // [max_frames, 1536]

  // Fast scratch buffer for weights unpacking (16-byte aligned)
  void *w_scratch_raw;
  float *w_scratch;

  // Cached FP32 weights in PSRAM for ultra-fast attentive pooling & aggregation
  float *pool_l1_w_fp32;
  float *pool_l1_x_fp32;
  float *pool_l2_w_fp32;
  float *conv_agg_w_fp32;

  // Working buffers (stored in engine to prevent stack overflow on RISC-V!)
  float se_mean[512];
  float se1[128];
  float se2[512];
  float se_scale[512];
  float u_mean[1536];
  float s_std[1536];
  float pooled[3072];
  float pooled_bn[3072];
  float raw_emb[192];

  // TDNN-Tiny working buffers
  float tdnn_stats[1600];
  float tdnn_seg1[512];
  float tdnn_raw_emb[512];
};

esp_err_t sr_engine_create(const sr_model_t *model, size_t max_frames,
                           bool use_psram, sr_engine_t **out_engine) {
  if (!model || !out_engine || max_frames == 0) {
    return ESP_ERR_INVALID_ARG;
  }

  sr_engine_t *engine = (sr_engine_t *)calloc(1, sizeof(sr_engine_t));
  if (!engine) {
    return ESP_ERR_NO_MEM;
  }

  engine->model = *model;
  engine->max_frames = max_frames;
  engine->is_fp16 = (model->quant_type == SR_QUANT_FP16);

  uint32_t caps = use_psram ? (MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)
                            : (MALLOC_CAP_DEFAULT | MALLOC_CAP_8BIT);

  engine->buf_512_A =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);
  engine->buf_512_B =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);
  engine->buf_512_C =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);
  engine->buf_512_x2 =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);
  engine->buf_512_x3 =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);
  engine->buf_512_x4 =
      (float *)heap_caps_malloc(max_frames * 512 * sizeof(float), caps);

  engine->buf_branch_in =
      (float *)heap_caps_malloc(max_frames * 64 * sizeof(float), caps);
  engine->buf_branch_out =
      (float *)heap_caps_malloc(max_frames * 64 * sizeof(float), caps);

  engine->buf_1536_cat =
      (float *)heap_caps_malloc(max_frames * 1536 * sizeof(float), caps);
  engine->buf_1536_agg =
      (float *)heap_caps_malloc(max_frames * 1536 * sizeof(float), caps);

  // Attentive pooling: att_in and att_h1 only need 1 frame (allocated in fast
  // L1 SRAM)
  engine->buf_att_in = (float *)heap_caps_malloc(
      4608 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!engine->buf_att_in)
    engine->buf_att_in = (float *)malloc(4608 * sizeof(float));

  engine->buf_att_h1 = (float *)heap_caps_malloc(
      max_frames * 128 * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!engine->buf_att_h1)
    engine->buf_att_h1 = (float *)heap_caps_malloc(max_frames * 128 * sizeof(float), caps);

  engine->buf_att_h2 =
      (float *)heap_caps_malloc(max_frames * 1536 * sizeof(float), caps);

  void *w_raw = heap_caps_malloc(
      SR_SIMD_SCRATCH_FLOATS * sizeof(float) + 32, caps);
  if (!w_raw) {
    w_raw = malloc(SR_SIMD_SCRATCH_FLOATS * sizeof(float) + 32);
  }
  engine->w_scratch_raw = w_raw;
  engine->w_scratch = w_raw ? (float *)(((uintptr_t)w_raw + 15) & ~15) : NULL;

  if (!engine->buf_512_A || !engine->buf_512_B || !engine->buf_512_x2 ||
      !engine->buf_512_x3 || !engine->buf_512_x4 || !engine->buf_branch_in ||
      !engine->buf_branch_out || !engine->buf_1536_cat ||
      !engine->buf_1536_agg || !engine->buf_att_in || !engine->buf_att_h1 ||
      !engine->buf_att_h2 || !engine->w_scratch) {
    ESP_LOGE(TAG, "Failed to allocate ECAPA-TDNN activation buffers in PSRAM");
    sr_engine_destroy(engine);
    return ESP_ERR_NO_MEM;
  }

  engine->pool_l1_w_fp32 = NULL;
  engine->pool_l1_x_fp32 = NULL;
  engine->pool_l2_w_fp32 = NULL;
  engine->conv_agg_w_fp32 = NULL;
  if (model->pool_dtype == SR_DTYPE_INT8) {
    if (use_psram) {
      engine->pool_l1_w_fp32 = (float *)heap_caps_malloc(128 * 4608 * sizeof(float), caps);
      if (engine->pool_l1_w_fp32) {
        const int8_t *src = (const int8_t *)model->pool_l1_weight;
        for (int o = 0; o < 128; o++) {
          float scale = model->pool_l1_scales ? model->pool_l1_scales[o] : 1.0f;
          for (int i = 0; i < 4608; i++) {
            engine->pool_l1_w_fp32[o * 4608 + i] = (float)src[o * 4608 + i] * scale;
          }
        }
        ESP_LOGI(TAG, "Cached INT8 pool_l1_weight (2.25 MB) in PSRAM");
      }
      engine->pool_l1_x_fp32 = (float *)heap_caps_malloc(128 * 1536 * sizeof(float), caps);
      if (engine->pool_l1_x_fp32 && engine->pool_l1_w_fp32) {
        for (int o = 0; o < 128; o++) {
          memcpy(engine->pool_l1_x_fp32 + o * 1536, engine->pool_l1_w_fp32 + o * 4608, 1536 * sizeof(float));
        }
        ESP_LOGI(TAG, "Cached pool_l1_x_weight (0.75 MB) contiguous in PSRAM");
      }
      engine->pool_l2_w_fp32 = (float *)heap_caps_malloc(1536 * 128 * sizeof(float), caps);
      if (engine->pool_l2_w_fp32) {
        const int8_t *src = (const int8_t *)model->pool_l2_weight;
        for (int o = 0; o < 1536; o++) {
          float scale = model->pool_l2_scales ? model->pool_l2_scales[o] : 1.0f;
          for (int i = 0; i < 128; i++) {
            engine->pool_l2_w_fp32[o * 128 + i] = (float)src[o * 128 + i] * scale;
          }
        }
        ESP_LOGI(TAG, "Cached INT8 pool_l2_weight (0.75 MB) in PSRAM");
      }
    }
  } else if (model->pool_dtype == SR_DTYPE_FP16) {
    if (use_psram) {
      engine->pool_l1_w_fp32 = (float *)heap_caps_malloc(128 * 4608 * sizeof(float), caps);
      if (engine->pool_l1_w_fp32) {
        sr_simd_unpack_fp16((const uint16_t *)model->pool_l1_weight, engine->pool_l1_w_fp32, 128 * 4608);
        ESP_LOGI(TAG, "Cached pool_l1_weight (2.25 MB) in PSRAM");
      }
      engine->pool_l1_x_fp32 = (float *)heap_caps_malloc(128 * 1536 * sizeof(float), caps);
      if (engine->pool_l1_x_fp32 && engine->pool_l1_w_fp32) {
        for (int o = 0; o < 128; o++) {
          memcpy(engine->pool_l1_x_fp32 + o * 1536, engine->pool_l1_w_fp32 + o * 4608, 1536 * sizeof(float));
        }
        ESP_LOGI(TAG, "Cached pool_l1_x_weight (0.75 MB) contiguous in PSRAM");
      }
      engine->pool_l2_w_fp32 = (float *)heap_caps_malloc(1536 * 128 * sizeof(float), caps);
      if (engine->pool_l2_w_fp32) {
        sr_simd_unpack_fp16((const uint16_t *)model->pool_l2_weight, engine->pool_l2_w_fp32, 1536 * 128);
        ESP_LOGI(TAG, "Cached pool_l2_weight (0.75 MB) in PSRAM");
      }
    }
  } else {
    engine->pool_l1_w_fp32 = (float *)model->pool_l1_weight;
    engine->pool_l1_x_fp32 = (float *)heap_caps_malloc(128 * 1536 * sizeof(float), caps);
    if (engine->pool_l1_x_fp32 && engine->pool_l1_w_fp32) {
      for (int o = 0; o < 128; o++) {
        memcpy(engine->pool_l1_x_fp32 + o * 1536, engine->pool_l1_w_fp32 + o * 4608, 1536 * sizeof(float));
      }
    }
    engine->pool_l2_w_fp32 = (float *)model->pool_l2_weight;
  }

  size_t total_alloc =
      max_frames * (512 * 5 + 64 * 2 + 1536 * 3) * sizeof(float) +
      (4608 + 128) * sizeof(float) +
      (engine->pool_l1_w_fp32 ? (128 * 4608 + 1536 * 128) * sizeof(float) : 0) +
      (engine->conv_agg_w_fp32 ? (1536 * 1536) * sizeof(float) : 0);
  ESP_LOGI(TAG,
           "ECAPA-TDNN Engine initialized: %.2f MB activation memory in %s",
           (float)total_alloc / (1024.0f * 1024.0f),
           use_psram ? "PSRAM" : "Internal SRAM");

  *out_engine = engine;
  return ESP_OK;
}

void sr_engine_destroy(sr_engine_t *engine) {
  if (!engine)
    return;
  if (engine->buf_512_A)
    free(engine->buf_512_A);
  if (engine->buf_512_B)
    free(engine->buf_512_B);
  if (engine->buf_512_C)
    free(engine->buf_512_C);
  if (engine->buf_512_x2)
    free(engine->buf_512_x2);
  if (engine->buf_512_x3)
    free(engine->buf_512_x3);
  if (engine->buf_512_x4)
    free(engine->buf_512_x4);
  if (engine->buf_branch_in)
    free(engine->buf_branch_in);
  if (engine->buf_branch_out)
    free(engine->buf_branch_out);
  if (engine->buf_1536_cat)
    free(engine->buf_1536_cat);
  if (engine->buf_1536_agg)
    free(engine->buf_1536_agg);
  if (engine->buf_att_in)
    free(engine->buf_att_in);
  if (engine->buf_att_h1)
    free(engine->buf_att_h1);
  if (engine->buf_att_h2)
    free(engine->buf_att_h2);
  if (engine->w_scratch_raw)
    free(engine->w_scratch_raw);
  else if (engine->w_scratch)
    free(engine->w_scratch);
  if (engine->pool_l1_w_fp32)
    free(engine->pool_l1_w_fp32);
  if (engine->pool_l1_x_fp32)
    free(engine->pool_l1_x_fp32);
  if (engine->pool_l2_w_fp32)
    free(engine->pool_l2_w_fp32);
  if (engine->conv_agg_w_fp32)
    free(engine->conv_agg_w_fp32);
  free(engine);
}

static void forward_se_res2block(sr_engine_t *engine, const float *in_feature,
                                 float *out_feature,
                                 const sr_se_res2block_t *block, int dilation,
                                 size_t num_frames) {
  float *h1 = engine->buf_512_B;
  float *res2 = engine->buf_512_C;
  float *h2 = engine->buf_512_B; // Reuse buffer B

  // 1. conv0 (512 -> 512, 1x1) + ReLU + BN
  sr_simd_conv1d(in_feature, h1, block->conv0.weight, block->conv0.bias,
                 block->conv0.bn_scale, block->conv0.bn_bias,
                 block->conv0.w_scales, engine->w_scratch,
                 512, 512, 1, 1, 0, (int)num_frames, block->conv0.dtype, true);

  // 2. Res2Net 7 branches (kernel 3, dilation, pad=dilation, 64 -> 64)
  float *b_in = engine->buf_branch_in;
  float *b_out = engine->buf_branch_out;

  for (int b = 0; b < 7; b++) {
    // Prepare branch input
    if (b == 0) {
      for (size_t t = 0; t < num_frames; t++) {
        memcpy(b_in + t * 64, h1 + t * 512 + 0 * 64, 64 * sizeof(float));
      }
    } else {
      // b_in = chunk_b + prev_branch_out
      for (size_t t = 0; t < num_frames; t++) {
        const float *chunk = h1 + t * 512 + b * 64;
        const float *prev = b_out + t * 64;
        float *dst = b_in + t * 64;
        for (int i = 0; i < 64; i++) {
          dst[i] = chunk[i] + prev[i];
        }
      }
    }

    // Branch conv + ReLU + BN
    sr_simd_conv1d(b_in, b_out, block->branches[b].weight,
                   block->branches[b].bias, block->branches[b].bn_scale,
                   block->branches[b].bn_bias, block->branches[b].w_scales,
                   engine->w_scratch, 64, 64, 3,
                   dilation, dilation, (int)num_frames, block->branches[b].dtype, true);

    // Store branch output into res2 slice b
    for (size_t t = 0; t < num_frames; t++) {
      memcpy(res2 + t * 512 + b * 64, b_out + t * 64, 64 * sizeof(float));
    }
  }

  // Chunk 7 is passed through identity
  for (size_t t = 0; t < num_frames; t++) {
    memcpy(res2 + t * 512 + 7 * 64, h1 + t * 512 + 7 * 64, 64 * sizeof(float));
  }

  // 3. conv2 (512 -> 512, 1x1) + ReLU + BN
  sr_simd_conv1d(res2, h2, block->conv2.weight, block->conv2.bias,
                 block->conv2.bn_scale, block->conv2.bn_bias,
                 block->conv2.w_scales, engine->w_scratch,
                 512, 512, 1, 1, 0, (int)num_frames, block->conv2.dtype, true);

  // 4. Squeeze-and-Excitation
  float *se_mean = engine->se_mean;
  for (int c = 0; c < 512; c++) {
    float sum = 0.0f;
    for (size_t t = 0; t < num_frames; t++) {
      sum += h2[t * 512 + c];
    }
    se_mean[c] = sum / (float)num_frames;
  }

  float *se1 = engine->se1;
  sr_simd_gemm_mv(se_mean, se1, block->se_l1_weight, block->se_l1_bias,
                  block->se_l1_scales, engine->w_scratch, 512, 128, block->se_dtype, true);

  float *se2 = engine->se2;
  sr_simd_gemm_mv(se1, se2, block->se_l2_weight, block->se_l2_bias,
                  block->se_l2_scales, engine->w_scratch, 128, 512, block->se_dtype, false);

  float *se_scale = engine->se_scale;
  for (int c = 0; c < 512; c++) {
    se_scale[c] = 1.0f / (1.0f + expf(-se2[c]));
  }

  // Multiply scale and add residual: out = in + h2 * scale
  for (size_t t = 0; t < num_frames; t++) {
    const float *in_r = in_feature + t * 512;
    const float *h2_r = h2 + t * 512;
    float *out_r = out_feature + t * 512;
    for (int c = 0; c < 512; c++) {
      out_r[c] = in_r[c] + h2_r[c] * se_scale[c];
    }
  }
}

// ============================================================================
// Attentive Statistics Pooling Dual-Core Jobs
// ============================================================================
typedef struct {
    sr_engine_t *engine;
    const float *base_bias;
    const float *w_p1_x;
    size_t num_frames;
} att_l1_job_t;

static void att_l1_part(void *ctx, int part_idx, int total_parts) {
    att_l1_job_t *job = (att_l1_job_t *)ctx;
    sr_engine_t *engine = job->engine;
    const float *w_p1_x = job->w_p1_x;
    size_t num_frames = job->num_frames;
    float *h1 = engine->buf_att_h1;

    // Partition 128 output channels across cores
    int i_start_all = (128 * part_idx) / total_parts;
    int i_end_all = (128 * (part_idx + 1)) / total_parts;

    // Each core uses half of w_scratch (16384 floats = 64 KB SRAM)
    float *my_scratch = engine->w_scratch + part_idx * 16384;

    // Block channels into chunks of 8 (8 * 1536 floats = 12,288 floats = 48 KB)
    for (int i_start = i_start_all; i_start < i_end_all; i_start += 8) {
        int i_end = (i_start + 8 < i_end_all) ? (i_start + 8) : i_end_all;
        int cur_block = i_end - i_start;

        // Copy weights into fast L1 internal SRAM once
        memcpy(my_scratch, w_p1_x + i_start * 1536, cur_block * 1536 * sizeof(float));

        // Stream through frames t
        for (size_t t = 0; t < num_frames; t++) {
            const float *x_t = engine->buf_1536_agg + t * 1536;
            float *h1_t = h1 + t * 128;
            for (int b = 0; b < cur_block; b++) {
                int i = i_start + b;
                const float *w_row = my_scratch + b * 1536;
                float dot = 0.0f;
                dsps_dotprod_f32(x_t, w_row, &dot, 1536);
                h1_t[i] = tanhf(dot + job->base_bias[i]);
            }
        }
    }
}

typedef struct {
    sr_engine_t *engine;
    const float *w_p2;
    size_t num_frames;
} att_l2_reduce_job_t;

static void att_l2_reduce_part(void *ctx, int part_idx, int total_parts) {
    att_l2_reduce_job_t *job = (att_l2_reduce_job_t *)ctx;
    sr_engine_t *engine = job->engine;
    const float *w_p2 = job->w_p2;
    size_t num_frames = job->num_frames;
    float *h1 = engine->buf_att_h1;
    float *pooled = engine->pooled;

    // Partition 1536 channels across cores
    int c_start = (1536 * part_idx) / total_parts;
    int c_end = (1536 * (part_idx + 1)) / total_parts;

    // Local buffer for channel attention logits over time (stack allocated, max 320 floats = 1.25 KB)
    float h2_local[320];
    if (num_frames > 320) return;

    for (int c = c_start; c < c_end; c++) {
        const float *w_row = w_p2 + c * 128;
        float b = engine->model.pool_l2_bias ? engine->model.pool_l2_bias[c] : 0.0f;

        // 1. Compute attention logit for this channel across all frames t
        float max_val = -1e30f;
        for (size_t t = 0; t < num_frames; t++) {
            const float *h1_t = h1 + t * 128;
            float dot = 0.0f;
            dsps_dotprod_f32(h1_t, w_row, &dot, 128);
            float v = dot + b;
            h2_local[t] = v;
            if (v > max_val) max_val = v;
        }

        // 2. Softmax normalization over time
        float sum_exp = 0.0f;
        for (size_t t = 0; t < num_frames; t++) {
            float v = expf(h2_local[t] - max_val);
            h2_local[t] = v;
            sum_exp += v;
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);

        // 3. Weighted mean & weighted square mean across time
        float w_mean = 0.0f;
        float w_sq_mean = 0.0f;
        for (size_t t = 0; t < num_frames; t++) {
            float alpha = h2_local[t] * inv_sum;
            float x_val = engine->buf_1536_agg[t * 1536 + c];
            w_mean += alpha * x_val;
            w_sq_mean += alpha * (x_val * x_val);
        }

        // 4. Weighted standard deviation
        float w_var = w_sq_mean - w_mean * w_mean;
        if (w_var < 1e-7f) w_var = 1e-7f;
        float w_std = sqrtf(w_var);

        pooled[c] = w_mean;
        pooled[1536 + c] = w_std;
    }
}

int sr_engine_get_embedding_dim(const sr_engine_t *engine) {
  if (!engine) return 192;
  return engine->model.embedding_dim ? engine->model.embedding_dim : 192;
}

static esp_err_t sr_engine_infer_tdnn_tiny(sr_engine_t *engine, const float *mel_input,
                                           size_t num_frames, float *out_embedding) {
  int64_t t0 = esp_timer_get_time();

  // Conv1: [num_frames, 80] -> [T1, 256], K=5, dil=1, pad=0, ReLU
  int t1_frames = (int)num_frames - 4;
  if (t1_frames <= 0) return ESP_ERR_INVALID_SIZE;

  sr_simd_conv1d(mel_input, engine->buf_512_A,
                 engine->model.tdnn_conv1.weight, engine->model.tdnn_conv1.bias,
                 NULL, NULL, engine->model.tdnn_conv1.w_scales, engine->w_scratch,
                 80, 256, 5, 1, 0, (int)num_frames,
                 engine->model.tdnn_conv1.dtype, true);
  int64_t t1 = esp_timer_get_time();

  // Conv2: [T1, 256] -> [T2, 256], K=3, dil=2, pad=0, ReLU
  int t2_frames = t1_frames - 4;
  if (t2_frames <= 0) return ESP_ERR_INVALID_SIZE;

  sr_simd_conv1d(engine->buf_512_A, engine->buf_512_B,
                 engine->model.tdnn_conv2.weight, engine->model.tdnn_conv2.bias,
                 NULL, NULL, engine->model.tdnn_conv2.w_scales, engine->w_scratch,
                 256, 256, 3, 2, 0, t1_frames,
                 engine->model.tdnn_conv2.dtype, true);
  int64_t t2 = esp_timer_get_time();

  // Conv3: [T2, 256] -> [T3, 256], K=3, dil=3, pad=0, ReLU
  int t3_frames = t2_frames - 6;
  if (t3_frames <= 0) return ESP_ERR_INVALID_SIZE;

  sr_simd_conv1d(engine->buf_512_B, engine->buf_512_A,
                 engine->model.tdnn_conv3.weight, engine->model.tdnn_conv3.bias,
                 NULL, NULL, engine->model.tdnn_conv3.w_scales, engine->w_scratch,
                 256, 256, 3, 3, 0, t2_frames,
                 engine->model.tdnn_conv3.dtype, true);
  int64_t t3 = esp_timer_get_time();

  // Conv4: [T3, 256] -> [T4, 256], K=1, dil=1, pad=0, ReLU
  sr_simd_conv1d(engine->buf_512_A, engine->buf_512_B,
                 engine->model.tdnn_conv4.weight, engine->model.tdnn_conv4.bias,
                 NULL, NULL, engine->model.tdnn_conv4.w_scales, engine->w_scratch,
                 256, 256, 1, 1, 0, t3_frames,
                 engine->model.tdnn_conv4.dtype, true);
  int64_t t4 = esp_timer_get_time();

  // Conv5: [T4, 256] -> [T5, 800], K=1, dil=1, pad=0, ReLU
  sr_simd_conv1d(engine->buf_512_B, engine->buf_1536_cat,
                 engine->model.tdnn_conv5.weight, engine->model.tdnn_conv5.bias,
                 NULL, NULL, engine->model.tdnn_conv5.w_scales, engine->w_scratch,
                 256, 800, 1, 1, 0, t3_frames,
                 engine->model.tdnn_conv5.dtype, true);
  int64_t t5 = esp_timer_get_time();

  // Stats Pooling: mean and sample std (ddof=1) across time frames for 800 channels
  float *stats = engine->tdnn_stats;
  float bessel = (t3_frames > 1) ? ((float)t3_frames / (float)(t3_frames - 1)) : 1.0f;
  for (int c = 0; c < 800; c++) {
    float sum = 0.0f;
    for (int t = 0; t < t3_frames; t++) {
      sum += engine->buf_1536_cat[t * 800 + c];
    }
    float m = sum / (float)t3_frames;
    stats[c] = m;

    float sq_diff = 0.0f;
    for (int t = 0; t < t3_frames; t++) {
      float d = engine->buf_1536_cat[t * 800 + c] - m;
      sq_diff += d * d;
    }
    float var_val = (sq_diff / (float)t3_frames) * bessel;
    stats[800 + c] = sqrtf(var_val > 0.0f ? var_val : 0.0f);
  }
  int64_t t6 = esp_timer_get_time();

  // Gemm 1: Linear [1600 -> 512], ReLU
  float *seg1 = engine->tdnn_seg1;
  sr_simd_gemm_mv(stats, seg1, engine->model.tdnn_fc1_weight, engine->model.tdnn_fc1_bias,
                  engine->model.tdnn_fc1_scales, engine->w_scratch,
                  1600, 512, engine->model.tdnn_fc1_dtype, true);

  // Gemm 2: Linear [512 -> 512], Linear
  float *raw_emb = engine->tdnn_raw_emb;
  sr_simd_gemm_mv(seg1, raw_emb, engine->model.tdnn_fc2_weight, engine->model.tdnn_fc2_bias,
                  engine->model.tdnn_fc2_scales, engine->w_scratch,
                  512, 512, engine->model.tdnn_fc2_dtype, false);
  int64_t t7 = esp_timer_get_time();

  // L2 Normalization
  float sum_sq = 0.0f;
  for (int i = 0; i < 512; i++) {
    sum_sq += raw_emb[i] * raw_emb[i];
  }
  float norm = sqrtf(sum_sq + 1e-12f);
  float inv_norm = 1.0f / norm;
  for (int i = 0; i < 512; i++) {
    out_embedding[i] = raw_emb[i] * inv_norm;
  }

  ESP_LOGI(TAG, "TDNN-Tiny Real-Time Profile (%zu frames -> %d pooled): Conv1=%.1fms, Conv2=%.1fms, Conv3=%.1fms, Conv4=%.1fms, Conv5=%.1fms, Pool=%.1fms, FC=%.1fms, Total=%.1fms",
           num_frames, t3_frames,
           (float)(t1-t0)/1000.0f, (float)(t2-t1)/1000.0f, (float)(t3-t2)/1000.0f,
           (float)(t4-t3)/1000.0f, (float)(t5-t4)/1000.0f, (float)(t6-t5)/1000.0f,
           (float)(t7-t6)/1000.0f, (float)(t7-t0)/1000.0f);

  return ESP_OK;
}

esp_err_t sr_engine_infer(sr_engine_t *engine, const float *mel_input,
                          size_t num_frames, float *out_embedding) {
  if (!engine || !mel_input || !out_embedding || num_frames == 0) {
    return ESP_ERR_INVALID_ARG;
  }
  if (num_frames > engine->max_frames) {
    ESP_LOGE(TAG, "num_frames (%zu) exceeds max_frames (%zu)", num_frames,
             engine->max_frames);
    return ESP_ERR_INVALID_SIZE;
  }

  if (engine->model.arch == SR_ARCH_TDNN_TINY) {
    return sr_engine_infer_tdnn_tiny(engine, mel_input, num_frames, out_embedding);
  }

  (void)engine;

  int64_t t0 = esp_timer_get_time();

  // 1. Layer 1: Conv1D (80 -> 512, K=5, pad=2) + ReLU + BN
  sr_simd_conv1d(mel_input, engine->buf_512_A, engine->model.layer1_conv.weight,
                 engine->model.layer1_conv.bias,
                 engine->model.layer1_conv.bn_scale,
                 engine->model.layer1_conv.bn_bias,
                 engine->model.layer1_conv.w_scales, engine->w_scratch, 80, 512,
                 5, 1, 2, (int)num_frames, engine->model.layer1_conv.dtype, true);
  int64_t t1 = esp_timer_get_time();

  // 2. Layer 2, 3, 4: SE-Res2Blocks
  forward_se_res2block(engine, engine->buf_512_A, engine->buf_512_x2,
                       &engine->model.layer2, 2, num_frames);
  int64_t t2 = esp_timer_get_time();

  forward_se_res2block(engine, engine->buf_512_x2, engine->buf_512_x3,
                       &engine->model.layer3, 3, num_frames);
  int64_t t3 = esp_timer_get_time();

  forward_se_res2block(engine, engine->buf_512_x3, engine->buf_512_x4,
                       &engine->model.layer4, 4, num_frames);
  int64_t t4 = esp_timer_get_time();

  // 3. Multi-layer concatenation: [x2, x3, x4] -> [num_frames, 1536]
  for (size_t t = 0; t < num_frames; t++) {
    memcpy(engine->buf_1536_cat + t * 1536 + 0, engine->buf_512_x2 + t * 512,
           512 * sizeof(float));
    memcpy(engine->buf_1536_cat + t * 1536 + 512, engine->buf_512_x3 + t * 512,
           512 * sizeof(float));
    memcpy(engine->buf_1536_cat + t * 1536 + 1024, engine->buf_512_x4 + t * 512,
           512 * sizeof(float));
  }

  // 4. Aggregation Conv (1536 -> 1536, 1x1) + ReLU
  const void *w_agg = engine->conv_agg_w_fp32 ? (const void *)engine->conv_agg_w_fp32 : engine->model.conv_agg.weight;
  uint8_t agg_dtype = engine->conv_agg_w_fp32 ? 4 /* FP32 */ : engine->model.conv_agg.dtype;
  ESP_LOGI(TAG, "Agg conv config: dtype=%u, w=%p, scales=%p", (unsigned)agg_dtype, w_agg, engine->model.conv_agg.w_scales);
  sr_simd_conv1d(engine->buf_1536_cat, engine->buf_1536_agg,
                 w_agg, engine->model.conv_agg.bias,
                 NULL, NULL, engine->model.conv_agg.w_scales, engine->w_scratch, 1536, 1536, 1, 1, 0,
                 (int)num_frames, agg_dtype, true);
  int64_t t5 = esp_timer_get_time();

  // 5. Attentive Statistics Pooling
  // Compute global mean and std across time for each of 1536 channels
  float *u_mean = engine->u_mean;
  float *s_std = engine->s_std;
  float bessel =
      (num_frames > 1) ? ((float)num_frames / (float)(num_frames - 1)) : 1.0f;

  for (int c = 0; c < 1536; c++) {
    float sum = 0.0f;
    for (size_t t = 0; t < num_frames; t++) {
      sum += engine->buf_1536_agg[t * 1536 + c];
    }
    float m = sum / (float)num_frames;
    u_mean[c] = m;

    float sq_diff = 0.0f;
    for (size_t t = 0; t < num_frames; t++) {
      float d = engine->buf_1536_agg[t * 1536 + c] - m;
      sq_diff += d * d;
    }
    float var_val = (sq_diff / (float)num_frames) * bessel + 1e-5f;
    s_std[c] = sqrtf(var_val);
  }

  // 5. Attentive Statistics Pooling
  // Step 5a: Precompute base_bias[128] for Layer 1 invariants (u_mean and s_std)
  float base_bias[128];
  const float *w_p1 = engine->pool_l1_w_fp32;
  const float *w_p1_x = engine->pool_l1_x_fp32;
  const float *w_p2 = engine->pool_l2_w_fp32;

  for (int i = 0; i < 128; i++) {
    const float *w_row = w_p1 + i * 4608;
    float b = engine->model.pool_l1_bias ? engine->model.pool_l1_bias[i] : 0.0f;
    float dot_u = 0.0f;
    float dot_s = 0.0f;
    dsps_dotprod_f32(u_mean, w_row + 1536, &dot_u, 1536);
    dsps_dotprod_f32(s_std, w_row + 3072, &dot_s, 1536);
    base_bias[i] = b + dot_u + dot_s;
  }

  // Step 5b: Dual-Core Layer 1 (blocked weights in fast SRAM)
  att_l1_job_t l1_job = {
    .engine = engine,
    .base_bias = base_bias,
    .w_p1_x = w_p1_x,
    .num_frames = num_frames
  };
  sr_parallel_run(att_l1_part, &l1_job);

  // Step 5c: Dual-Core Online Softmax + Reduce (zero PSRAM allocation for H2)
  att_l2_reduce_job_t l2_job = {
    .engine = engine,
    .w_p2 = w_p2,
    .num_frames = num_frames
  };
  sr_parallel_run(att_l2_reduce_part, &l2_job);

  // 6. Output BN (3072)
  int64_t t6 = esp_timer_get_time();
  float *pooled = engine->pooled;
  float *pooled_bn = engine->pooled_bn;
  for (int i = 0; i < 3072; i++) {
    pooled_bn[i] = pooled[i] * engine->model.out_bn_scale[i] +
                   engine->model.out_bn_bias[i];
  }

  // 7. Output Linear (3072 -> 192)
  float *raw_emb = engine->raw_emb;
  sr_simd_gemm_mv(pooled_bn, raw_emb, engine->model.out_l_weight,
                  engine->model.out_l_bias, engine->model.out_l_scales,
                  engine->w_scratch, 3072, 192,
                  engine->model.out_l_dtype, false);
  int64_t t7 = esp_timer_get_time();

  ESP_LOGI(TAG, "Inference profile (%zu frames): L1=%.1fms, L2=%.1fms, L3=%.1fms, L4=%.1fms, Agg=%.1fms, Pool=%.1fms, Out=%.1fms, Total=%.1fms",
           num_frames, (float)(t1-t0)/1000.0f, (float)(t2-t1)/1000.0f, (float)(t3-t2)/1000.0f,
           (float)(t4-t3)/1000.0f, (float)(t5-t4)/1000.0f, (float)(t6-t5)/1000.0f, (float)(t7-t6)/1000.0f, (float)(t7-t0)/1000.0f);

  // 8. Subtract centering vector (mean_vec)
  for (int i = 0; i < 192; i++) {
    raw_emb[i] -= engine->model.mean_vec[i];
  }

  // 9. L2 Normalization
  float sum_sq = 0.0f;
  for (int i = 0; i < 192; i++) {
    sum_sq += raw_emb[i] * raw_emb[i];
  }
  float norm = sqrtf(sum_sq + 1e-12f);
  float inv_norm = 1.0f / norm;

  for (int i = 0; i < 192; i++) {
    out_embedding[i] = raw_emb[i] * inv_norm;
  }

  return ESP_OK;
}
