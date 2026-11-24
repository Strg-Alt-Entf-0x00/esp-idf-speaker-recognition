#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hardware-accelerated 1D INT8 Dot Product via ESP32-P4 RV32 PIE SIMD
 * Computes 16 signed 8-bit MACs per cycle using hardware accumulator.
 */
esp_err_t sr_dotprod_s8_pie(const int8_t *src1, const int8_t *src2, int32_t *dest, int len);

#define SR_SIMD_SCRATCH_FLOATS 32768

/**
 * @brief Ultra-fast 1D Convolution over contiguous [T, in_c] feature maps
 *
 * Implements: out[t, c] = BN(ReLU(sum_{k} dot(x[t + k*d - p], W[c, k]) + bias[c]))
 * Weights are stored contiguously as [out_c, K, in_c].
 */
void sr_simd_conv1d(
    const float *input,
    float *output,
    const void *weight,
    const float *bias,
    const float *bn_scale,
    const float *bn_bias,
    const float *w_scales,
    float *w_scratch,
    int in_c,
    int out_c,
    int kernel_size,
    int dilation,
    int pad,
    int num_frames,
    uint8_t dtype,
    bool apply_relu
);

/**
 * @brief Dense Matrix-Vector product: out[M] = weight[M, N] * in[N] + bias[M]
 *
 * @param in_vec Input vector [in_dim]
 * @param out_vec Output vector [out_dim]
 * @param weight Weight matrix [out_dim, in_dim]
 * @param bias Bias vector [out_dim]
 * @param w_scales Per-channel scale vector for INT8 weights
 * @param w_scratch Scratch buffer for unpacking
 * @param in_dim Input dimension
 * @param out_dim Output dimension
 * @param dtype Data type (SR_DTYPE_INT8, SR_DTYPE_FP16, SR_DTYPE_FP32)
 * @param apply_relu Apply ReLU activation
 */
void sr_simd_gemm_mv(
    const float *in_vec,
    float *out_vec,
    const void *weight,
    const float *bias,
    const float *w_scales,
    float *w_scratch,
    int in_dim,
    int out_dim,
    uint8_t dtype,
    bool apply_relu
);

/**
 * @brief Unpack FP16 half-precision weights to FP32 single-precision
 */
void sr_simd_unpack_fp16(const uint16_t *src, float *dst, size_t count);

/**
 * @brief Vector elementwise addition: a += b
 */
void sr_simd_vector_add(
    float *a,
    const float *b,
    size_t count
);

/**
 * @brief Vector dot product for cosine similarity
 */
float sr_simd_dot_product(
    const float *vec1,
    const float *vec2,
    size_t dim
);

#ifdef __cplusplus
}
#endif
