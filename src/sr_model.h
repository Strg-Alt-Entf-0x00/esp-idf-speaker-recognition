#pragma once

#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SR_ESPM_MAGIC          0x4D505345  // 'ESPM'
#define SR_ESPM_VERSION        3
#define SR_MODEL_TYPE_SPEAKER  4

#define SR_DTYPE_INT8   1
#define SR_DTYPE_INT16  2
#define SR_DTYPE_FP16   3
#define SR_DTYPE_FP32   4

#define SR_QUANT_INT8_PER_CH 2
#define SR_QUANT_FP16        3
#define SR_QUANT_FP32        4

typedef enum {
    SR_ARCH_UNKNOWN = 0,
    SR_ARCH_ECAPA_TDNN = 1,
    SR_ARCH_CAMPP = 2,
    SR_ARCH_TDNN_TINY = 3,
} sr_model_arch_t;

typedef struct {
    const void *weight;         // Pointer to weight data (INT8, FP16 or FP32)
    const float *bias;          // Pointer to FP32 bias
    const float *bn_scale;      // Fused BatchNorm scale: gamma / sqrt(var + eps)
    const float *bn_bias;       // Fused BatchNorm bias: beta - mean * scale
    const float *w_scales;      // Per-channel INT8 scales: w_real = w_int8 * w_scales[c]
    uint8_t dtype;              // Weight data type (SR_DTYPE_INT8, SR_DTYPE_FP16, SR_DTYPE_FP32)
    uint16_t out_channels;
    uint16_t in_channels;
    uint8_t kernel_size;
} sr_conv1d_t;

typedef struct {
    sr_conv1d_t conv0;          // [512, 512, 1] + BN
    sr_conv1d_t branches[7];    // 7 branches: [64, 64, 3] + BN
    sr_conv1d_t conv2;          // [512, 512, 1] + BN
    // Squeeze-and-Excitation
    const void *se_l1_weight;   // [128, 512]
    const float *se_l1_bias;    // [128]
    const float *se_l1_scales;  // [128]
    const void *se_l2_weight;   // [512, 128]
    const float *se_l2_bias;    // [512]
    const float *se_l2_scales;  // [512]
    uint8_t se_dtype;
} sr_se_res2block_t;

typedef struct {
    sr_model_arch_t arch;
    uint8_t quant_type;
    uint32_t sample_rate;
    uint16_t embedding_dim;     // 192
    uint16_t mel_bins;          // 80

    // Layer 1: Conv1D (512, 80, 5) + BN
    sr_conv1d_t layer1_conv;

    // Layers 2, 3, 4: SE-Res2Blocks
    sr_se_res2block_t layer2;
    sr_se_res2block_t layer3;
    sr_se_res2block_t layer4;

    // Multi-layer aggregation conv (1536, 1536, 1)
    sr_conv1d_t conv_agg;

    // Attentive Statistics Pooling
    const void *pool_l1_weight; // [128, 4608]
    const float *pool_l1_bias;  // [128]
    const float *pool_l1_scales;// [128]
    const void *pool_l2_weight; // [1536, 128]
    const float *pool_l2_bias;  // [1536]
    const float *pool_l2_scales;// [1536]
    uint8_t pool_dtype;

    // Output BN (3072) and Linear (192, 3072)
    const float *out_bn_scale;  // [3072]
    const float *out_bn_bias;   // [3072]
    const void *out_l_weight;   // [192, 3072]
    const float *out_l_bias;    // [192]
    const float *out_l_scales;  // [192]
    uint8_t out_l_dtype;

    // Speaker centering vector (192)
    const float *mean_vec;      // [192]

    // TDNN-Tiny Layers (Real-Time Profile)
    sr_conv1d_t tdnn_conv1;     // [256, 80, 5], dil=1, pad=0
    sr_conv1d_t tdnn_conv2;     // [256, 256, 3], dil=2, pad=0
    sr_conv1d_t tdnn_conv3;     // [256, 256, 3], dil=3, pad=0
    sr_conv1d_t tdnn_conv4;     // [256, 256, 1], dil=1, pad=0
    sr_conv1d_t tdnn_conv5;     // [800, 256, 1], dil=1, pad=0
    const void *tdnn_fc1_weight; // [512, 1600]
    const float *tdnn_fc1_bias;  // [512]
    const float *tdnn_fc1_scales;// [512]
    uint8_t tdnn_fc1_dtype;
    const void *tdnn_fc2_weight; // [512, 512]
    const float *tdnn_fc2_bias;  // [512]
    const float *tdnn_fc2_scales;// [512]
    uint8_t tdnn_fc2_dtype;

    // Unpacked FP32 weights in PSRAM (NULL if mmapped directly)
    void *unpacked_weights_psram;
} sr_model_t;

/**
 * @brief Parse and validate ESPM v3 Speaker Recognition model from memory buffer
 *
 * @param buffer Pointer to raw ESPM file in memory
 * @param buffer_size Size in bytes
 * @param out_model Output model structure with resolved layer pointers
 * @return ESP_OK on success
 */
esp_err_t sr_model_load(const uint8_t *buffer, size_t buffer_size, sr_model_t *out_model);

/**
 * @brief Unpack FP16 weights into a contiguous FP32 buffer in PSRAM for maximum inference speed
 *
 * @param model Model structure to unpack in-place
 * @return ESP_OK on success
 */
esp_err_t sr_model_unpack_to_psram(sr_model_t *model);

/**
 * @brief Free unpacked PSRAM weights if allocated
 *
 * @param model Model structure
 */
void sr_model_free_psram_weights(sr_model_t *model);

#ifdef __cplusplus
}
#endif
