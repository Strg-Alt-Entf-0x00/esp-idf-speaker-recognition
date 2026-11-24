#include "sr_model.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_rom_crc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "SR_MODEL";

#define ESPM_SECTION_INDEX 6
#define ESPM_SECTION_DATA 7

typedef struct __attribute__((packed)) {
  uint32_t magic;
  uint16_t version;
  uint16_t model_type;
  uint16_t quant_type;
  uint16_t num_sections;
  char model_name[32];
  char version_str[16];
  uint32_t sample_rate;
  uint32_t psram_required_kb;
  uint32_t header_crc;
  uint64_t file_size;
  uint32_t header_size;
  uint16_t section_offset;
  uint16_t section_entry_size;
  uint32_t alignment;
  uint32_t flags;
  uint32_t section_table_crc;
  uint8_t reserved[28];
} espm_header_t;

typedef struct __attribute__((packed)) {
  uint16_t kind;
  uint16_t flags;
  uint64_t offset;
  uint64_t size;
  uint32_t crc32;
  char name[16];
} espm_section_entry_t;

typedef struct __attribute__((packed)) {
  char magic[4]; // "ETIX"
  uint16_t version;
  uint16_t header_size;
  uint16_t record_size;
  uint16_t flags;
  uint32_t num_tensors;
  uint32_t num_partitions;
  uint64_t total_elements;
  uint64_t total_bytes;
} espm_index_header_t;

typedef struct __attribute__((packed)) {
  uint32_t name_hash;
  uint32_t flags;
  uint8_t dtype;
  uint8_t ndim;
  uint8_t quant_mode;
  uint8_t reserved0;
  uint32_t num_elements;
  uint32_t dims[8];
  uint32_t part_id;
  uint8_t reserved1;
  uint8_t reserved2[3];
  uint64_t crc32;
  uint32_t reserved3;
  uint64_t data_offset;
  uint64_t data_size;
  char name[64];
  uint8_t reserved4[12];
} espm_tensor_entry_t;

_Static_assert(sizeof(espm_header_t) == 128, "espm_header_t size must be 128");
_Static_assert(sizeof(espm_section_entry_t) == 40,
               "espm_section_entry_t size must be 40");
_Static_assert(sizeof(espm_index_header_t) == 36,
               "espm_index_header_t size must be 36");
_Static_assert(sizeof(espm_tensor_entry_t) == 160,
               "espm_tensor_entry_t size must be 160");

typedef struct {
  const void *data;
  uint32_t size;
  uint8_t dtype;
  uint8_t ndim;
  uint32_t dims[8];
} parsed_tensor_t;

#define MAX_PARSED_TENSORS 256
static parsed_tensor_t s_parsed[MAX_PARSED_TENSORS];
static char s_parsed_names[MAX_PARSED_TENSORS][64];
static size_t s_num_parsed = 0;

static const parsed_tensor_t *find_tensor(const char *name) {
  for (size_t i = 0; i < s_num_parsed; i++) {
    if (strcmp(s_parsed_names[i], name) == 0) {
      return &s_parsed[i];
    }
  }
  return NULL;
}

static esp_err_t resolve_conv(const char *w_name, const char *b_name,
                              const char *bn_s_name, const char *bn_b_name,
                              sr_conv1d_t *layer) {
  const parsed_tensor_t *t_w = find_tensor(w_name);
  const parsed_tensor_t *t_b = find_tensor(b_name);
  if (!t_w || !t_b) {
    ESP_LOGE(TAG, "Missing weights for conv: %s / %s", w_name, b_name);
    return ESP_ERR_NOT_FOUND;
  }
  layer->weight = t_w->data;
  layer->bias = (const float *)t_b->data;
  layer->dtype = t_w->dtype;

  // Resolve INT8 per-channel scales if present
  layer->w_scales = NULL;
  char scale_name[96];
  snprintf(scale_name, sizeof(scale_name), "%s.scale", w_name);
  const parsed_tensor_t *t_scale = find_tensor(scale_name);
  if (t_scale) {
    layer->w_scales = (const float *)t_scale->data;
  }

  if (t_w->ndim == 3) {
    layer->out_channels = (uint16_t)t_w->dims[0];
    layer->kernel_size = (uint8_t)t_w->dims[1];
    layer->in_channels = (uint16_t)t_w->dims[2];
  } else if (t_w->ndim == 2) {
    layer->out_channels = (uint16_t)t_w->dims[0];
    layer->kernel_size = 1;
    layer->in_channels = (uint16_t)t_w->dims[1];
  }

  if (bn_s_name && bn_b_name) {
    const parsed_tensor_t *t_s = find_tensor(bn_s_name);
    const parsed_tensor_t *t_bn_b = find_tensor(bn_b_name);
    if (t_s && t_bn_b) {
      layer->bn_scale = (const float *)t_s->data;
      layer->bn_bias = (const float *)t_bn_b->data;
    }
  }
  return ESP_OK;
}

static inline float fp16_to_fp32_fast(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000) << 16;
  int32_t e = (h >> 10) & 0x1F;
  uint32_t m = h & 0x03FF;
  uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = s;
    } else {
      while (!(m & 0x0400)) {
        m <<= 1;
        e--;
      }
      e++;
      m &= ~0x0400;
      u = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) {
    u = s | 0x7F800000 | (m << 13);
  } else {
    u = s | ((uint32_t)(e + 112) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &u, sizeof(f));
  return f;
}

esp_err_t sr_model_unpack_to_psram(sr_model_t *model) {
  if (!model)
    return ESP_ERR_INVALID_ARG;
  if (model->quant_type != SR_QUANT_FP16) {
    return ESP_OK; // already FP32
  }

  // Sort indices by tensor size descending to prevent heap fragmentation
  size_t order[MAX_PARSED_TENSORS];
  for (size_t i = 0; i < s_num_parsed; i++)
    order[i] = i;

  for (size_t i = 0; i < s_num_parsed; i++) {
    for (size_t j = i + 1; j < s_num_parsed; j++) {
      if (s_parsed[order[j]].size > s_parsed[order[i]].size) {
        size_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
    }
  }

  size_t total_elements = 0;
  for (size_t idx = 0; idx < s_num_parsed; idx++) {
    size_t i = order[idx];
    if (s_parsed[i].dtype == SR_DTYPE_FP16) {
      size_t count = s_parsed[i].size / 2;
      total_elements += count;

      const uint16_t *src = (const uint16_t *)s_parsed[i].data;
      float *dst = (float *)heap_caps_malloc(
          count * sizeof(float), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      if (!dst) {
        ESP_LOGE(
            TAG,
            "Failed to allocate %zu bytes in PSRAM for tensor '%s' (largest "
            "free block: %u MB)",
            count * sizeof(float), s_parsed_names[i],
            (unsigned)(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM) /
                       (1024 * 1024)));
        return ESP_ERR_NO_MEM;
      }

      for (size_t j = 0; j < count; j++) {
        dst[j] = fp16_to_fp32_fast(src[j]);
      }

      s_parsed[i].data = dst;
      s_parsed[i].dtype = SR_DTYPE_FP32;
      s_parsed[i].size = count * sizeof(float);
    }
  }

  ESP_LOGI(TAG, "Unpacked %zu FP16 weights (%.2f MB) into PSRAM successfully!",
           total_elements,
           (float)(total_elements * sizeof(float)) / (1024.0f * 1024.0f));

  // Re-resolve all model pointers to point to new FP32 PSRAM weights
  resolve_conv("layer1.conv.weight", "layer1.conv.bias", "layer1.bn.scale",
               "layer1.bn.bias", &model->layer1_conv);

  char w_name[64], b_name[64], s_name[64], bn_b_name[64];
  for (int l = 2; l <= 4; l++) {
    sr_se_res2block_t *block =
        (l == 2) ? &model->layer2
                 : ((l == 3) ? &model->layer3 : &model->layer4);
    snprintf(w_name, sizeof(w_name), "layer%d.conv0.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.conv0.bias", l);
    snprintf(s_name, sizeof(s_name), "layer%d.conv0.bn_scale", l);
    snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.conv0.bn_bias", l);
    resolve_conv(w_name, b_name, s_name, bn_b_name, &block->conv0);

    for (int b = 0; b < 7; b++) {
      snprintf(w_name, sizeof(w_name), "layer%d.branch%d.weight", l, b);
      snprintf(b_name, sizeof(b_name), "layer%d.branch%d.bias", l, b);
      snprintf(s_name, sizeof(s_name), "layer%d.branch%d.bn_scale", l, b);
      snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.branch%d.bn_bias", l, b);
      resolve_conv(w_name, b_name, s_name, bn_b_name, &block->branches[b]);
    }

    snprintf(w_name, sizeof(w_name), "layer%d.conv2.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.conv2.bias", l);
    snprintf(s_name, sizeof(s_name), "layer%d.conv2.bn_scale", l);
    snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.conv2.bn_bias", l);
    resolve_conv(w_name, b_name, s_name, bn_b_name, &block->conv2);

    snprintf(w_name, sizeof(w_name), "layer%d.se.linear1.weight", l);
    const parsed_tensor_t *t_se1_w = find_tensor(w_name);
    if (t_se1_w) {
      block->se_l1_weight = t_se1_w->data;
      block->se_dtype = t_se1_w->dtype;
    }

    snprintf(w_name, sizeof(w_name), "layer%d.se.linear2.weight", l);
    const parsed_tensor_t *t_se2_w = find_tensor(w_name);
    if (t_se2_w) {
      block->se_l2_weight = t_se2_w->data;
    }
  }

  resolve_conv("conv_agg.weight", "conv_agg.bias", NULL, NULL,
               &model->conv_agg);

  const parsed_tensor_t *p_l1_w = find_tensor("pool.linear1.weight");
  if (p_l1_w) {
    model->pool_l1_weight = p_l1_w->data;
    model->pool_dtype = p_l1_w->dtype;
  }
  const parsed_tensor_t *p_l2_w = find_tensor("pool.linear2.weight");
  if (p_l2_w) {
    model->pool_l2_weight = p_l2_w->data;
  }

  const parsed_tensor_t *out_lw = find_tensor("out.linear.weight");
  if (out_lw) {
    model->out_l_weight = out_lw->data;
    model->out_l_dtype = out_lw->dtype;
  }

  model->quant_type = SR_QUANT_FP32;
  ESP_LOGI(TAG, "FP16 weights unpacked to FP32 in PSRAM successfully!");
  return ESP_OK;
}

void sr_model_free_psram_weights(sr_model_t *model) {
  if (model && model->unpacked_weights_psram) {
    free(model->unpacked_weights_psram);
    model->unpacked_weights_psram = NULL;
  }
}

esp_err_t sr_model_load(const uint8_t *buffer, size_t buffer_size,
                        sr_model_t *out_model) {
  if (!buffer || buffer_size < sizeof(espm_header_t) || !out_model) {
    return ESP_ERR_INVALID_ARG;
  }

  memset(out_model, 0, sizeof(sr_model_t));
  const espm_header_t *hdr = (const espm_header_t *)buffer;

  if (hdr->magic != SR_ESPM_MAGIC) {
    ESP_LOGE(TAG, "Invalid magic: 0x%08lX (expected 0x%08X)",
             (unsigned long)hdr->magic, SR_ESPM_MAGIC);
    return ESP_ERR_INVALID_STATE;
  }
  if (hdr->version != SR_ESPM_VERSION) {
    ESP_LOGE(TAG, "Unsupported format version: %u", hdr->version);
    return ESP_ERR_NOT_SUPPORTED;
  }
  if (hdr->model_type != SR_MODEL_TYPE_SPEAKER) {
    ESP_LOGE(TAG, "Invalid model type: %u", hdr->model_type);
    return ESP_ERR_INVALID_STATE;
  }

  out_model->arch = SR_ARCH_ECAPA_TDNN;
  out_model->quant_type = (uint8_t)hdr->quant_type;
  out_model->sample_rate = hdr->sample_rate ? hdr->sample_rate : 16000;
  out_model->embedding_dim = 192;
  out_model->mel_bins = 80;

  ESP_LOGI(
      TAG,
      "Loading ESPM: model='%s', ver='%s', quant=%u, sections=%u, size=%lu KB",
      hdr->model_name, hdr->version_str, hdr->quant_type, hdr->num_sections,
      (unsigned long)(hdr->file_size / 1024));

  const espm_section_entry_t *sections =
      (const espm_section_entry_t *)(buffer + hdr->header_size);
  const espm_section_entry_t *sec_index = NULL;
  const espm_section_entry_t *sec_data = NULL;

  for (uint16_t i = 0; i < hdr->num_sections; i++) {
    if (sections[i].kind == ESPM_SECTION_INDEX)
      sec_index = &sections[i];
    else if (sections[i].kind == ESPM_SECTION_DATA)
      sec_data = &sections[i];
  }

  if (!sec_index || !sec_data) {
    ESP_LOGE(TAG, "Missing index or data section in ESPM");
    return ESP_ERR_INVALID_STATE;
  }

  const uint8_t *index_ptr = buffer + sec_index->offset;
  const uint8_t *data_base = buffer + sec_data->offset;
  const espm_index_header_t *idx_hdr = (const espm_index_header_t *)index_ptr;

  if (memcmp(idx_hdr->magic, "ETIX", 4) != 0) {
    ESP_LOGE(TAG, "Invalid index header magic");
    return ESP_ERR_INVALID_STATE;
  }

  s_num_parsed = 0;
  const espm_tensor_entry_t *tensors =
      (const espm_tensor_entry_t *)(index_ptr + idx_hdr->header_size);

  for (uint32_t i = 0; i < idx_hdr->num_tensors; i++) {
    if (s_num_parsed >= MAX_PARSED_TENSORS)
      break;
    const espm_tensor_entry_t *te = &tensors[i];
    parsed_tensor_t *pt = &s_parsed[s_num_parsed];
    strncpy(s_parsed_names[s_num_parsed], te->name, 63);
    s_parsed_names[s_num_parsed][63] = '\0';

    pt->data = data_base + te->data_offset;
    pt->size = (uint32_t)te->data_size;
    pt->dtype = te->dtype;
    pt->ndim = te->ndim;
    for (int d = 0; d < 8; d++)
      pt->dims[d] = te->dims[d];
    s_num_parsed++;
  }

  ESP_LOGI(TAG, "Indexed %zu tensors successfully", s_num_parsed);

  // Check model architecture
  if (find_tensor("tdnn.conv1.weight") != NULL) {
    out_model->arch = SR_ARCH_TDNN_TINY;
    out_model->embedding_dim = 512;
    ESP_LOGI(TAG, "Resolving TDNN-Tiny (Real-Time Profile, 512-D)...");

    ESP_ERROR_CHECK(resolve_conv("tdnn.conv1.weight", "tdnn.conv1.bias", NULL, NULL, &out_model->tdnn_conv1));
    ESP_ERROR_CHECK(resolve_conv("tdnn.conv2.weight", "tdnn.conv2.bias", NULL, NULL, &out_model->tdnn_conv2));
    ESP_ERROR_CHECK(resolve_conv("tdnn.conv3.weight", "tdnn.conv3.bias", NULL, NULL, &out_model->tdnn_conv3));
    ESP_ERROR_CHECK(resolve_conv("tdnn.conv4.weight", "tdnn.conv4.bias", NULL, NULL, &out_model->tdnn_conv4));
    ESP_ERROR_CHECK(resolve_conv("tdnn.conv5.weight", "tdnn.conv5.bias", NULL, NULL, &out_model->tdnn_conv5));

    const parsed_tensor_t *t_fc1_w = find_tensor("tdnn.fc1.weight");
    const parsed_tensor_t *t_fc1_b = find_tensor("tdnn.fc1.bias");
    const parsed_tensor_t *t_fc1_s = find_tensor("tdnn.fc1.weight.scale");
    if (!t_fc1_w || !t_fc1_b) {
      ESP_LOGE(TAG, "Missing tdnn.fc1 tensors");
      return ESP_ERR_NOT_FOUND;
    }
    out_model->tdnn_fc1_weight = t_fc1_w->data;
    out_model->tdnn_fc1_bias = (const float *)t_fc1_b->data;
    out_model->tdnn_fc1_dtype = t_fc1_w->dtype;
    out_model->tdnn_fc1_scales = t_fc1_s ? (const float *)t_fc1_s->data : NULL;

    const parsed_tensor_t *t_fc2_w = find_tensor("tdnn.fc2.weight");
    const parsed_tensor_t *t_fc2_b = find_tensor("tdnn.fc2.bias");
    const parsed_tensor_t *t_fc2_s = find_tensor("tdnn.fc2.weight.scale");
    if (!t_fc2_w || !t_fc2_b) {
      ESP_LOGE(TAG, "Missing tdnn.fc2 tensors");
      return ESP_ERR_NOT_FOUND;
    }
    out_model->tdnn_fc2_weight = t_fc2_w->data;
    out_model->tdnn_fc2_bias = (const float *)t_fc2_b->data;
    out_model->tdnn_fc2_dtype = t_fc2_w->dtype;
    out_model->tdnn_fc2_scales = t_fc2_s ? (const float *)t_fc2_s->data : NULL;

    ESP_LOGI(TAG, "TDNN-Tiny model resolved and verified successfully!");
    return ESP_OK;
  }

  out_model->arch = SR_ARCH_ECAPA_TDNN;
  out_model->embedding_dim = 192;
  // Resolve ECAPA-TDNN512 tensors
  // 1. Layer 1
  ESP_ERROR_CHECK(resolve_conv("layer1.conv.weight", "layer1.conv.bias",
                               "layer1.bn.scale", "layer1.bn.bias",
                               &out_model->layer1_conv));

  // 2. Layers 2, 3, 4
  for (int l = 2; l <= 4; l++) {
    sr_se_res2block_t *block =
        (l == 2) ? &out_model->layer2
                 : ((l == 3) ? &out_model->layer3 : &out_model->layer4);
    char w_name[64], b_name[64], s_name[96], bn_b_name[64];

    // conv0
    snprintf(w_name, sizeof(w_name), "layer%d.conv0.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.conv0.bias", l);
    snprintf(s_name, sizeof(s_name), "layer%d.conv0.bn_scale", l);
    snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.conv0.bn_bias", l);
    ESP_ERROR_CHECK(
        resolve_conv(w_name, b_name, s_name, bn_b_name, &block->conv0));

    // 7 branches
    for (int b = 0; b < 7; b++) {
      snprintf(w_name, sizeof(w_name), "layer%d.branch%d.weight", l, b);
      snprintf(b_name, sizeof(b_name), "layer%d.branch%d.bias", l, b);
      snprintf(s_name, sizeof(s_name), "layer%d.branch%d.bn_scale", l, b);
      snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.branch%d.bn_bias", l, b);
      ESP_ERROR_CHECK(
          resolve_conv(w_name, b_name, s_name, bn_b_name, &block->branches[b]));
    }

    // conv2
    snprintf(w_name, sizeof(w_name), "layer%d.conv2.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.conv2.bias", l);
    snprintf(s_name, sizeof(s_name), "layer%d.conv2.bn_scale", l);
    snprintf(bn_b_name, sizeof(bn_b_name), "layer%d.conv2.bn_bias", l);
    ESP_ERROR_CHECK(
        resolve_conv(w_name, b_name, s_name, bn_b_name, &block->conv2));

    // SE linear 1 & 2
    snprintf(w_name, sizeof(w_name), "layer%d.se.linear1.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.se.linear1.bias", l);
    const parsed_tensor_t *t_se1_w = find_tensor(w_name);
    const parsed_tensor_t *t_se1_b = find_tensor(b_name);
    if (t_se1_w && t_se1_b) {
      block->se_l1_weight = t_se1_w->data;
      block->se_l1_bias = (const float *)t_se1_b->data;
      block->se_dtype = t_se1_w->dtype;
      snprintf(s_name, sizeof(s_name), "%s.scale", w_name);
      const parsed_tensor_t *t_s = find_tensor(s_name);
      block->se_l1_scales = t_s ? (const float *)t_s->data : NULL;
    }

    snprintf(w_name, sizeof(w_name), "layer%d.se.linear2.weight", l);
    snprintf(b_name, sizeof(b_name), "layer%d.se.linear2.bias", l);
    const parsed_tensor_t *t_se2_w = find_tensor(w_name);
    const parsed_tensor_t *t_se2_b = find_tensor(b_name);
    if (t_se2_w && t_se2_b) {
      block->se_l2_weight = t_se2_w->data;
      block->se_l2_bias = (const float *)t_se2_b->data;
      snprintf(s_name, sizeof(s_name), "%s.scale", w_name);
      const parsed_tensor_t *t_s = find_tensor(s_name);
      block->se_l2_scales = t_s ? (const float *)t_s->data : NULL;
    }
  }

  // 3. Multi-layer Aggregation Conv
  ESP_ERROR_CHECK(resolve_conv("conv_agg.weight", "conv_agg.bias", NULL, NULL,
                               &out_model->conv_agg));

  // 4. Attentive Statistics Pooling
  const parsed_tensor_t *p_l1_w = find_tensor("pool.linear1.weight");
  const parsed_tensor_t *p_l1_b = find_tensor("pool.linear1.bias");
  const parsed_tensor_t *p_l2_w = find_tensor("pool.linear2.weight");
  const parsed_tensor_t *p_l2_b = find_tensor("pool.linear2.bias");
  if (p_l1_w && p_l1_b && p_l2_w && p_l2_b) {
    out_model->pool_l1_weight = p_l1_w->data;
    out_model->pool_l1_bias = (const float *)p_l1_b->data;
    out_model->pool_l2_weight = p_l2_w->data;
    out_model->pool_l2_bias = (const float *)p_l2_b->data;
    out_model->pool_dtype = p_l1_w->dtype;
    const parsed_tensor_t *t_p1_s = find_tensor("pool.linear1.weight.scale");
    out_model->pool_l1_scales = t_p1_s ? (const float *)t_p1_s->data : NULL;
    const parsed_tensor_t *t_p2_s = find_tensor("pool.linear2.weight.scale");
    out_model->pool_l2_scales = t_p2_s ? (const float *)t_p2_s->data : NULL;
  }

  // 5. Output BN, Linear & Centering Vector
  const parsed_tensor_t *out_s = find_tensor("out.bn.scale");
  const parsed_tensor_t *out_b = find_tensor("out.bn.bias");
  const parsed_tensor_t *out_lw = find_tensor("out.linear.weight");
  const parsed_tensor_t *out_lb = find_tensor("out.linear.bias");
  const parsed_tensor_t *t_mean = find_tensor("mean_vec");

  if (out_s && out_b && out_lw && out_lb && t_mean) {
    out_model->out_bn_scale = (const float *)out_s->data;
    out_model->out_bn_bias = (const float *)out_b->data;
    out_model->out_l_weight = out_lw->data;
    out_model->out_l_bias = (const float *)out_lb->data;
    out_model->out_l_dtype = out_lw->dtype;
    out_model->mean_vec = (const float *)t_mean->data;
    const parsed_tensor_t *t_out_s = find_tensor("out.linear.weight.scale");
    out_model->out_l_scales = t_out_s ? (const float *)t_out_s->data : NULL;
  }

  ESP_LOGI(TAG, "ECAPA-TDNN512 model resolved and verified successfully!");
  return ESP_OK;
}
