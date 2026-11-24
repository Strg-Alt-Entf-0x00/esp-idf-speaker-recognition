#include "sr_simd_pie.h"
#include "sr_parallel.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_cpu.h"
#include "dsps_dotprod.h"
#include <math.h>
#include <string.h>

static const char *TAG = "SR_SIMD";

static inline float fp16_to_fp32(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16;
  int32_t e = (h >> 10) & 0x1F;
  uint32_t m = h & 0x03FF;
  uint32_t u;
  if (__builtin_expect(e > 0 && e < 31, 1)) {
    u = s | ((uint32_t)(e + 112) << 23) | (m << 13);
  } else if (e == 0) {
    u = (m == 0) ? s : 0;
  } else {
    u = s | 0x7F800000 | (m << 13);
  }
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

void sr_simd_unpack_fp16(const uint16_t *src, float *dst, size_t count) {
  for (size_t i = 0; i < count; i++) {
    dst[i] = fp16_to_fp32(src[i]);
  }
}

// ============================================================================
// Ultra-Fast INT8 1x1 Conv1D (Hardware RV32 PIE SIMD + Dual-Core Partitioning)
// ============================================================================
typedef struct {
  const int8_t *in_q;
  const float *x_scales;
  float *output;
  const int8_t *w_int8;
  const float *w_scales;
  const float *bias;
  const float *bn_scale;
  const float *bn_bias;
  float *w_scratch;
  int in_c;
  int out_c;
  int num_frames;
  bool has_bn;
  bool apply_relu;
} conv1d_1x1_int8_job_t;

static void conv1d_1x1_int8_part(void *ctx, int part_idx, int total_parts) {
  conv1d_1x1_int8_job_t *job = (conv1d_1x1_int8_job_t *)ctx;
  int c_part_start = (job->out_c * part_idx) / total_parts;
  int c_part_end = (job->out_c * (part_idx + 1)) / total_parts;
  int in_c = job->in_c;
  int num_frames = job->num_frames;

  // Each core uses half of w_scratch (16384 floats = 64 KB L1 SRAM)
  int8_t *w_sram = (int8_t *)(job->w_scratch + part_idx * 16384);

  int c_block = (48 * 1024) / in_c;
  if (c_block > 32) c_block = 32;
  if (c_block < 1) c_block = 1;
  if (c_block >= 8) c_block = (c_block / 8) * 8;

  for (int c_start = c_part_start; c_start < c_part_end; c_start += c_block) {
    int cur_block = c_part_end - c_start;
    if (cur_block > c_block) cur_block = c_block;

    // Copy weights ONCE from PSRAM into L1 SRAM for these cur_block channels!
    memcpy(w_sram, job->w_int8 + c_start * in_c, cur_block * in_c);

    // Pre-load channel parameters
    float b_val[32], w_sc[32], bn_s[32], bn_b[32];
    for (int b = 0; b < cur_block; b++) {
      int c = c_start + b;
      b_val[b] = job->bias ? job->bias[c] : 0.0f;
      w_sc[b] = job->w_scales ? job->w_scales[c] : 1.0f;
      bn_s[b] = job->has_bn ? job->bn_scale[c] : 1.0f;
      bn_b[b] = job->has_bn ? job->bn_bias[c] : 0.0f;
    }

    // Stream through frames
    for (int t = 0; t < num_frames; t++) {
      const int8_t *in_row = job->in_q + t * in_c;
      float *out_row = job->output + t * job->out_c;
      float x_scale = job->x_scales[t];

      // Compute dot products with hardware PIE INT8 SIMD
      for (int b = 0; b < cur_block; b++) {
        const int8_t *w_row = w_sram + b * in_c;
        int32_t sum_i32 = 0;
        sr_dotprod_s8_pie(in_row, w_row, &sum_i32, in_c);

        float val = (float)sum_i32 * (x_scale * w_sc[b]) + b_val[b];
        if (job->apply_relu && val < 0.0f) {
          val = 0.0f;
        }
        if (job->has_bn) {
          val = val * bn_s[b] + bn_b[b];
        }
        out_row[c_start + b] = val;
      }
    }
  }
}

// ============================================================================
// General INT8 Conv1D Dual-Core Job (K >= 1) with Hardware PIE SIMD
// ============================================================================
typedef struct {
  const int8_t *in_q;
  const float *x_scales;
  float *output;
  const int8_t *w_int8;
  const float *w_scales;
  const float *bias;
  const float *bn_scale;
  const float *bn_bias;
  float *w_scratch;
  int in_c;
  int out_c;
  int kernel_size;
  int dilation;
  int pad;
  int in_frames;
  int out_frames;
  bool has_bn;
  bool apply_relu;
} conv1d_k_int8_job_t;

static void conv1d_k_int8_part(void *ctx, int part_idx, int total_parts) {
  conv1d_k_int8_job_t *job = (conv1d_k_int8_job_t *)ctx;
  int c_part_start = (job->out_c * part_idx) / total_parts;
  int c_part_end = (job->out_c * (part_idx + 1)) / total_parts;
  int in_c = job->in_c;
  int K = job->kernel_size;
  int dilation = job->dilation;
  int pad = job->pad;
  int in_frames = job->in_frames;
  int out_frames = job->out_frames;
  int total_k_weights = K * in_c;

  // Each core uses half of w_scratch (16384 floats = 64 KB L1 SRAM)
  uintptr_t sram_addr = (uintptr_t)(job->w_scratch + part_idx * 16384);
  int8_t *w_sram = (int8_t *)((sram_addr + 15) & ~15);

  int c_block = (48 * 1024) / total_k_weights;
  if (c_block > 32) c_block = 32;
  if (c_block < 1) c_block = 1;
  if (c_block >= 8) c_block = (c_block / 8) * 8;

  for (int c_start = c_part_start; c_start < c_part_end; c_start += c_block) {
    int cur_block = c_part_end - c_start;
    if (cur_block > c_block) cur_block = c_block;

    // Copy weights ONCE from PSRAM into L1 SRAM for these cur_block channels!
    memcpy(w_sram, job->w_int8 + c_start * total_k_weights, cur_block * total_k_weights);

    // Pre-load channel parameters
    float b_val[32], w_sc[32], bn_s[32], bn_b[32];
    for (int b = 0; b < cur_block; b++) {
      int c = c_start + b;
      b_val[b] = job->bias ? job->bias[c] : 0.0f;
      w_sc[b] = job->w_scales ? job->w_scales[c] : 1.0f;
      bn_s[b] = job->has_bn ? job->bn_scale[c] : 1.0f;
      bn_b[b] = job->has_bn ? job->bn_bias[c] : 0.0f;
    }

    // Stream through output frames
    for (int t = 0; t < out_frames; t++) {
      float *out_row = job->output + t * job->out_c;

      for (int b = 0; b < cur_block; b++) {
        float acc = b_val[b];
        const int8_t *w_chan = w_sram + b * total_k_weights;

        for (int k = 0; k < K; k++) {
          int tin = t + k * dilation - pad;
          if (tin >= 0 && tin < in_frames) {
            const int8_t *in_row = job->in_q + tin * in_c;
            const int8_t *w_row = w_chan + k * in_c;
            int32_t sum_i32 = 0;
            sr_dotprod_s8_pie(in_row, w_row, &sum_i32, in_c);
            acc += (float)sum_i32 * (job->x_scales[tin] * w_sc[b]);
          }
        }

        if (job->apply_relu && acc < 0.0f) {
          acc = 0.0f;
        }
        if (job->has_bn) {
          acc = acc * bn_s[b] + bn_b[b];
        }
        out_row[c_start + b] = acc;
      }
    }
  }
}

// ============================================================================
// 1x1 Conv1D Dual-Core Job
// ============================================================================
typedef struct {
  const float *input;
  float *output;
  const float *bias;
  const float *bn_scale;
  const float *bn_bias;
  const float *w_scratch;
  int in_c;
  int out_c;
  int c_start;
  int cur_block;
  int num_frames;
  bool has_bn;
  bool apply_relu;
} conv1d_1x1_job_t;

static void conv1d_1x1_part(void *ctx, int part_idx, int total_parts) {
  conv1d_1x1_job_t *job = (conv1d_1x1_job_t *)ctx;
  int t_start = (job->num_frames * part_idx) / total_parts;
  int t_end = (job->num_frames * (part_idx + 1)) / total_parts;

  for (int t = t_start; t < t_end; t++) {
    const float *in_row = job->input + t * job->in_c;
    float *out_row = job->output + t * job->out_c;

    int b = 0;
    for (; b + 7 < job->cur_block; b += 8) {
      int c0 = job->c_start + b + 0;
      int c1 = job->c_start + b + 1;
      int c2 = job->c_start + b + 2;
      int c3 = job->c_start + b + 3;
      int c4 = job->c_start + b + 4;
      int c5 = job->c_start + b + 5;
      int c6 = job->c_start + b + 6;
      int c7 = job->c_start + b + 7;

      const float *w0 = job->w_scratch + (b + 0) * job->in_c;
      const float *w1 = job->w_scratch + (b + 1) * job->in_c;
      const float *w2 = job->w_scratch + (b + 2) * job->in_c;
      const float *w3 = job->w_scratch + (b + 3) * job->in_c;
      const float *w4 = job->w_scratch + (b + 4) * job->in_c;
      const float *w5 = job->w_scratch + (b + 5) * job->in_c;
      const float *w6 = job->w_scratch + (b + 6) * job->in_c;
      const float *w7 = job->w_scratch + (b + 7) * job->in_c;

      float acc0 = job->bias ? job->bias[c0] : 0.0f;
      float acc1 = job->bias ? job->bias[c1] : 0.0f;
      float acc2 = job->bias ? job->bias[c2] : 0.0f;
      float acc3 = job->bias ? job->bias[c3] : 0.0f;
      float acc4 = job->bias ? job->bias[c4] : 0.0f;
      float acc5 = job->bias ? job->bias[c5] : 0.0f;
      float acc6 = job->bias ? job->bias[c6] : 0.0f;
      float acc7 = job->bias ? job->bias[c7] : 0.0f;

      int k = 0;
      for (; k + 3 < job->in_c; k += 4) {
        float x0 = in_row[k + 0];
        float x1 = in_row[k + 1];
        float x2 = in_row[k + 2];
        float x3 = in_row[k + 3];

        acc0 +=
            x0 * w0[k + 0] + x1 * w0[k + 1] + x2 * w0[k + 2] + x3 * w0[k + 3];
        acc1 +=
            x0 * w1[k + 0] + x1 * w1[k + 1] + x2 * w1[k + 2] + x3 * w1[k + 3];
        acc2 +=
            x0 * w2[k + 0] + x1 * w2[k + 1] + x2 * w2[k + 2] + x3 * w2[k + 3];
        acc3 +=
            x0 * w3[k + 0] + x1 * w3[k + 1] + x2 * w3[k + 2] + x3 * w3[k + 3];
        acc4 +=
            x0 * w4[k + 0] + x1 * w4[k + 1] + x2 * w4[k + 2] + x3 * w4[k + 3];
        acc5 +=
            x0 * w5[k + 0] + x1 * w5[k + 1] + x2 * w5[k + 2] + x3 * w5[k + 3];
        acc6 +=
            x0 * w6[k + 0] + x1 * w6[k + 1] + x2 * w6[k + 2] + x3 * w6[k + 3];
        acc7 +=
            x0 * w7[k + 0] + x1 * w7[k + 1] + x2 * w7[k + 2] + x3 * w7[k + 3];
      }
      for (; k < job->in_c; k++) {
        float x = in_row[k];
        acc0 += x * w0[k];
        acc1 += x * w1[k];
        acc2 += x * w2[k];
        acc3 += x * w3[k];
        acc4 += x * w4[k];
        acc5 += x * w5[k];
        acc6 += x * w6[k];
        acc7 += x * w7[k];
      }

      if (job->apply_relu) {
        if (acc0 < 0.0f)
          acc0 = 0.0f;
        if (acc1 < 0.0f)
          acc1 = 0.0f;
        if (acc2 < 0.0f)
          acc2 = 0.0f;
        if (acc3 < 0.0f)
          acc3 = 0.0f;
        if (acc4 < 0.0f)
          acc4 = 0.0f;
        if (acc5 < 0.0f)
          acc5 = 0.0f;
        if (acc6 < 0.0f)
          acc6 = 0.0f;
        if (acc7 < 0.0f)
          acc7 = 0.0f;
      }
      if (job->has_bn) {
        acc0 = acc0 * job->bn_scale[c0] + job->bn_bias[c0];
        acc1 = acc1 * job->bn_scale[c1] + job->bn_bias[c1];
        acc2 = acc2 * job->bn_scale[c2] + job->bn_bias[c2];
        acc3 = acc3 * job->bn_scale[c3] + job->bn_bias[c3];
        acc4 = acc4 * job->bn_scale[c4] + job->bn_bias[c4];
        acc5 = acc5 * job->bn_scale[c5] + job->bn_bias[c5];
        acc6 = acc6 * job->bn_scale[c6] + job->bn_bias[c6];
        acc7 = acc7 * job->bn_scale[c7] + job->bn_bias[c7];
      }
      out_row[c0] = acc0;
      out_row[c1] = acc1;
      out_row[c2] = acc2;
      out_row[c3] = acc3;
      out_row[c4] = acc4;
      out_row[c5] = acc5;
      out_row[c6] = acc6;
      out_row[c7] = acc7;
    }

    for (; b + 3 < job->cur_block; b += 4) {
      int c0 = job->c_start + b + 0;
      int c1 = job->c_start + b + 1;
      int c2 = job->c_start + b + 2;
      int c3 = job->c_start + b + 3;

      const float *w0 = job->w_scratch + (b + 0) * job->in_c;
      const float *w1 = job->w_scratch + (b + 1) * job->in_c;
      const float *w2 = job->w_scratch + (b + 2) * job->in_c;
      const float *w3 = job->w_scratch + (b + 3) * job->in_c;

      float acc0 = job->bias ? job->bias[c0] : 0.0f;
      float acc1 = job->bias ? job->bias[c1] : 0.0f;
      float acc2 = job->bias ? job->bias[c2] : 0.0f;
      float acc3 = job->bias ? job->bias[c3] : 0.0f;

      int k = 0;
      for (; k + 3 < job->in_c; k += 4) {
        float x0 = in_row[k + 0];
        float x1 = in_row[k + 1];
        float x2 = in_row[k + 2];
        float x3 = in_row[k + 3];

        acc0 +=
            x0 * w0[k + 0] + x1 * w0[k + 1] + x2 * w0[k + 2] + x3 * w0[k + 3];
        acc1 +=
            x0 * w1[k + 0] + x1 * w1[k + 1] + x2 * w1[k + 2] + x3 * w1[k + 3];
        acc2 +=
            x0 * w2[k + 0] + x1 * w2[k + 1] + x2 * w2[k + 2] + x3 * w2[k + 3];
        acc3 +=
            x0 * w3[k + 0] + x1 * w3[k + 1] + x2 * w3[k + 2] + x3 * w3[k + 3];
      }
      for (; k < job->in_c; k++) {
        float x = in_row[k];
        acc0 += x * w0[k];
        acc1 += x * w1[k];
        acc2 += x * w2[k];
        acc3 += x * w3[k];
      }

      if (job->apply_relu) {
        if (acc0 < 0.0f)
          acc0 = 0.0f;
        if (acc1 < 0.0f)
          acc1 = 0.0f;
        if (acc2 < 0.0f)
          acc2 = 0.0f;
        if (acc3 < 0.0f)
          acc3 = 0.0f;
      }
      if (job->has_bn) {
        acc0 = acc0 * job->bn_scale[c0] + job->bn_bias[c0];
        acc1 = acc1 * job->bn_scale[c1] + job->bn_bias[c1];
        acc2 = acc2 * job->bn_scale[c2] + job->bn_bias[c2];
        acc3 = acc3 * job->bn_scale[c3] + job->bn_bias[c3];
      }
      out_row[c0] = acc0;
      out_row[c1] = acc1;
      out_row[c2] = acc2;
      out_row[c3] = acc3;
    }

    for (; b < job->cur_block; b++) {
      int c = job->c_start + b;
      const float *w_row = job->w_scratch + b * job->in_c;
      float acc = job->bias ? job->bias[c] : 0.0f;

      int k = 0;
      for (; k + 3 < job->in_c; k += 4) {
        acc += in_row[k + 0] * w_row[k + 0] + in_row[k + 1] * w_row[k + 1] +
               in_row[k + 2] * w_row[k + 2] + in_row[k + 3] * w_row[k + 3];
      }
      for (; k < job->in_c; k++) {
        acc += in_row[k] * w_row[k];
      }

      if (job->apply_relu && acc < 0.0f) {
        acc = 0.0f;
      }
      if (job->has_bn) {
        acc = acc * job->bn_scale[c] + job->bn_bias[c];
      }
      out_row[c] = acc;
    }
  }
}

// ============================================================================
// General Conv1D Dual-Core Job (K > 1)
// ============================================================================
typedef struct {
  const float *input;
  float *output;
  const float *bias;
  const float *bn_scale;
  const float *bn_bias;
  const float *w_scratch;
  int in_c;
  int out_c;
  int kernel_size;
  int dilation;
  int pad;
  int in_frames;
  int out_frames;
  int c_start;
  int cur_block;
  bool has_bn;
  bool apply_relu;
} conv1d_k_job_t;

static void conv1d_k_part(void *ctx, int part_idx, int total_parts) {
  conv1d_k_job_t *job = (conv1d_k_job_t *)ctx;
  int t_start = (job->out_frames * part_idx) / total_parts;
  int t_end = (job->out_frames * (part_idx + 1)) / total_parts;

  for (int t = t_start; t < t_end; t++) {
    float *out_row = job->output + t * job->out_c;

    for (int b = 0; b < job->cur_block; b++) {
      int c = job->c_start + b;
      float acc = job->bias ? job->bias[c] : 0.0f;

      for (int k = 0; k < job->kernel_size; k++) {
        int tin = t + k * job->dilation - job->pad;
        if (tin >= 0 && tin < job->in_frames) {
          const float *in_row = job->input + tin * job->in_c;
          const float *w_row =
              job->w_scratch + (b * job->kernel_size + k) * job->in_c;

          float dot = 0.0f;
          dsps_dotprod_f32(in_row, w_row, &dot, job->in_c);
          acc += dot;
        }
      }

      if (job->apply_relu && acc < 0.0f) {
        acc = 0.0f;
      }
      if (job->has_bn) {
        acc = acc * job->bn_scale[c] + job->bn_bias[c];
      }
      out_row[c] = acc;
    }
  }
}

// ============================================================================
// Public Conv1D Dispatcher
// ============================================================================
void sr_simd_conv1d(const float *input, float *output, const void *weight,
                    const float *bias, const float *bn_scale,
                    const float *bn_bias, const float *w_scales,
                    float *w_scratch, int in_c, int out_c,
                    int kernel_size, int dilation, int pad, int num_frames,
                    uint8_t dtype, bool apply_relu) {
  const int8_t *w_int8 = (const int8_t *)weight;
  const uint16_t *w_fp16 = (const uint16_t *)weight;
  const float *w_fp32 = (const float *)weight;
  bool has_bn = (bn_scale != NULL && bn_bias != NULL);
  int total_k_weights = kernel_size * in_c;

  int out_frames = (pad == 0) ? (num_frames - (kernel_size - 1) * dilation) : num_frames;
  if (out_frames < 1) out_frames = 1;

  // Ultra-Fast Path: INT8 Conv1D with Hardware RV32 PIE SIMD & Dual-Core (1x1 and K > 1)
  if (dtype == 1 /* SR_DTYPE_INT8 */ && (in_c % 16 == 0) && in_c <= 1536) {
    void *in_q_raw = heap_caps_malloc(num_frames * in_c + 32, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *in_q = in_q_raw ? (int8_t *)(((uintptr_t)in_q_raw + 15) & ~15) : NULL;
    float *x_scales = (float *)malloc(num_frames * sizeof(float));

    if (in_q && x_scales) {
      // Pre-quantize all input frames once
      for (int t = 0; t < num_frames; t++) {
        const float *in_row = input + t * in_c;
        int8_t *q_row = in_q + t * in_c;

        float max_v = 0.0f;
        for (int i = 0; i < in_c; i++) {
          float v = fabsf(in_row[i]);
          if (v > max_v) max_v = v;
        }
        x_scales[t] = max_v * (1.0f / 127.0f);
        float inv_scale = (max_v > 1e-8f) ? (127.0f / max_v) : 0.0f;
        for (int i = 0; i < in_c; i++) {
          float q = in_row[i] * inv_scale;
          int val = (int)(q >= 0.0f ? (q + 0.5f) : (q - 0.5f));
          if (val > 127) val = 127;
          else if (val < -127) val = -127;
          q_row[i] = (int8_t)val;
        }
      }

      if (kernel_size == 1 && dilation == 1 && pad == 0) {
        conv1d_1x1_int8_job_t job = {
          .in_q = in_q,
          .x_scales = x_scales,
          .output = output,
          .w_int8 = w_int8,
          .w_scales = w_scales,
          .bias = bias,
          .bn_scale = bn_scale,
          .bn_bias = bn_bias,
          .w_scratch = w_scratch,
          .in_c = in_c,
          .out_c = out_c,
          .num_frames = num_frames,
          .has_bn = has_bn,
          .apply_relu = apply_relu
        };
        sr_parallel_run(conv1d_1x1_int8_part, &job);
      } else {
        conv1d_k_int8_job_t job = {
          .in_q = in_q,
          .x_scales = x_scales,
          .output = output,
          .w_int8 = w_int8,
          .w_scales = w_scales,
          .bias = bias,
          .bn_scale = bn_scale,
          .bn_bias = bn_bias,
          .w_scratch = w_scratch,
          .in_c = in_c,
          .out_c = out_c,
          .kernel_size = kernel_size,
          .dilation = dilation,
          .pad = pad,
          .in_frames = num_frames,
          .out_frames = out_frames,
          .has_bn = has_bn,
          .apply_relu = apply_relu
        };
        sr_parallel_run(conv1d_k_int8_part, &job);
      }

      free(in_q_raw);
      free(x_scales);
      return;
    }
    if (in_q_raw) free(in_q_raw);
    if (x_scales) free(x_scales);
  }

  // Block channels so that weights fit into w_scratch (SR_SIMD_SCRATCH_FLOATS)
  int c_block = SR_SIMD_SCRATCH_FLOATS / total_k_weights;
  if (c_block >= 8) {
    c_block = (c_block / 8) * 8;
  }
  if (c_block < 1)
    c_block = 1;
  if (c_block > 64)
    c_block = 64;

  for (int c_start = 0; c_start < out_c; c_start += c_block) {
    int c_end = c_start + c_block;
    if (c_end > out_c)
      c_end = out_c;
    int cur_block = c_end - c_start;

    // Unpack cur_block channels into L1 SRAM
    if (dtype == 1 /* SR_DTYPE_INT8 */) {
      for (int b = 0; b < cur_block; b++) {
        int c = c_start + b;
        float scale = w_scales ? w_scales[c] : 1.0f;
        const int8_t *src = w_int8 + (c_start + b) * total_k_weights;
        float *dst = w_scratch + b * total_k_weights;
        for (int i = 0; i < total_k_weights; i++) {
          dst[i] = (float)src[i] * scale;
        }
      }
    } else if (dtype == 3 /* SR_DTYPE_FP16 */) {
      const uint16_t *src_w = w_fp16 + c_start * total_k_weights;
      sr_simd_unpack_fp16(src_w, w_scratch, cur_block * total_k_weights);
    } else {
      memcpy(w_scratch, w_fp32 + c_start * total_k_weights,
             cur_block * total_k_weights * sizeof(float));
    }

    if (kernel_size == 1 && dilation == 1 && pad == 0) {
      // Fast Path: 1x1 Conv1D on Dual Cores
      conv1d_1x1_job_t job = {.input = input,
                              .output = output,
                              .bias = bias,
                              .bn_scale = bn_scale,
                              .bn_bias = bn_bias,
                              .w_scratch = w_scratch,
                              .in_c = in_c,
                              .out_c = out_c,
                              .c_start = c_start,
                              .cur_block = cur_block,
                              .num_frames = num_frames,
                              .has_bn = has_bn,
                              .apply_relu = apply_relu};
      sr_parallel_run(conv1d_1x1_part, &job);
    } else {
      // General Path: K > 1 on Dual Cores
      out_frames = (pad == 0) ? (num_frames - (kernel_size - 1) * dilation) : num_frames;
      if (out_frames < 1) out_frames = 1;
      conv1d_k_job_t job = {.input = input,
                            .output = output,
                            .bias = bias,
                            .bn_scale = bn_scale,
                            .bn_bias = bn_bias,
                            .w_scratch = w_scratch,
                            .in_c = in_c,
                            .out_c = out_c,
                            .kernel_size = kernel_size,
                            .dilation = dilation,
                            .pad = pad,
                            .in_frames = num_frames,
                            .out_frames = out_frames,
                            .c_start = c_start,
                            .cur_block = cur_block,
                            .has_bn = has_bn,
                            .apply_relu = apply_relu};
      sr_parallel_run(conv1d_k_part, &job);
    }
  }
}

// ============================================================================
// Dense Matrix-Vector Product
// ============================================================================
void sr_simd_gemm_mv(const float *input, float *output, const void *weight,
                     const float *bias, const float *w_scales, float *w_scratch,
                     int in_dim, int out_dim, uint8_t dtype, bool apply_relu) {
  const int8_t *w_int8 = (const int8_t *)weight;
  const uint16_t *w_fp16 = (const uint16_t *)weight;
  const float *w_fp32 = (const float *)weight;

  // Ultra-Fast Path: INT8 Matrix-Vector via RV32 PIE SIMD
  if (dtype == 1 /* SR_DTYPE_INT8 */ && (in_dim % 16 == 0) && in_dim <= 4096) {
    size_t in_q_bytes = ((in_dim + 63) & ~63);
    int8_t *in_q = (int8_t *)w_scratch;
    int8_t *w_sram = (int8_t *)w_scratch + in_q_bytes;

    float max_v = 0.0f;
    for (int i = 0; i < in_dim; i++) {
      float v = fabsf(input[i]);
      if (v > max_v)
        max_v = v;
    }
    float x_scale = max_v * (1.0f / 127.0f);
    float inv_scale = (max_v > 1e-8f) ? (127.0f / max_v) : 0.0f;
    for (int i = 0; i < in_dim; i++) {
      float q = input[i] * inv_scale;
      int val = (int)roundf(q);
      if (val > 127)
        val = 127;
      else if (val < -127)
        val = -127;
      in_q[i] = (int8_t)val;
    }

    int o_block = ((SR_SIMD_SCRATCH_FLOATS * sizeof(float)) - in_q_bytes) / in_dim;
    if (o_block < 1)
      o_block = 1;
    if (o_block > 128)
      o_block = 128;

    for (int o_start = 0; o_start < out_dim; o_start += o_block) {
      int o_end = o_start + o_block;
      if (o_end > out_dim)
        o_end = out_dim;
      int cur_block = o_end - o_start;

      memcpy(w_sram, w_int8 + o_start * in_dim, cur_block * in_dim);

      for (int b = 0; b < cur_block; b++) {
        int o = o_start + b;
        int32_t dot_i32 = 0;
        sr_dotprod_s8_pie(in_q, w_sram + b * in_dim, &dot_i32, in_dim);

        float eff_scale = x_scale * (w_scales ? w_scales[o] : 1.0f);
        float acc = (float)dot_i32 * eff_scale + (bias ? bias[o] : 0.0f);
        if (apply_relu && acc < 0.0f)
          acc = 0.0f;
        output[o] = acc;
      }
    }
    return;
  }

  int o_block = 16000 / in_dim;
  if (o_block < 1)
    o_block = 1;
  if (o_block > 64)
    o_block = 64;

  for (int o_start = 0; o_start < out_dim; o_start += o_block) {
    int o_end = o_start + o_block;
    if (o_end > out_dim)
      o_end = out_dim;
    int cur_block = o_end - o_start;

    if (dtype == 1 /* SR_DTYPE_INT8 */) {
      for (int b = 0; b < cur_block; b++) {
        int o = o_start + b;
        float scale = w_scales ? w_scales[o] : 1.0f;
        const int8_t *src = w_int8 + (o_start + b) * in_dim;
        float *dst = w_scratch + b * in_dim;
        for (int i = 0; i < in_dim; i++) {
          dst[i] = (float)src[i] * scale;
        }
      }
    } else if (dtype == 3 /* SR_DTYPE_FP16 */) {
      const uint16_t *src_w = w_fp16 + o_start * in_dim;
      sr_simd_unpack_fp16(src_w, w_scratch, cur_block * in_dim);
    } else {
      memcpy(w_scratch, w_fp32 + o_start * in_dim,
             cur_block * in_dim * sizeof(float));
    }

    for (int b = 0; b < cur_block; b++) {
      int o = o_start + b;
      const float *w_row = w_scratch + b * in_dim;
      float acc = (bias ? bias[o] : 0.0f);

      int k = 0;
      for (; k + 3 < in_dim; k += 4) {
        acc += input[k + 0] * w_row[k + 0] + input[k + 1] * w_row[k + 1] +
               input[k + 2] * w_row[k + 2] + input[k + 3] * w_row[k + 3];
      }
      for (; k < in_dim; k++) {
        acc += input[k] * w_row[k];
      }

      if (apply_relu && acc < 0.0f) {
        acc = 0.0f;
      }
      output[o] = acc;
    }
  }
}

float sr_simd_dot_product(const float *a, const float *b, size_t len) {
  float dot = 0.0f;
  size_t i = 0;
  for (; i + 3 < len; i += 4) {
    dot += a[i + 0] * b[i + 0] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] +
           a[i + 3] * b[i + 3];
  }
  for (; i < len; i++) {
    dot += a[i] * b[i];
  }
  return dot;
}
