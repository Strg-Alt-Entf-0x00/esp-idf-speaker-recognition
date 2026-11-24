#pragma once

#include "esp_err.h"
#include "sr_model.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sr_engine sr_engine_t;

/**
 * @brief Create speaker recognition inference engine
 *
 * @param model Loaded model
 * @param max_frames Maximum audio frames to support (e.g. 350 for ~3.5s)
 * @param use_psram Whether to place activation buffers in PSRAM
 * @param out_engine Output engine pointer
 */
esp_err_t sr_engine_create(
    const sr_model_t *model,
    size_t max_frames,
    bool use_psram,
    sr_engine_t **out_engine
);

/**
 * @brief Destroy speaker recognition engine and free buffers
 */
void sr_engine_destroy(sr_engine_t *engine);

/**
 * @brief Get embedding dimension of loaded engine model (192 for ECAPA, 512 for TDNN-Tiny)
 */
int sr_engine_get_embedding_dim(const sr_engine_t *engine);

/**
 * @brief Run inference on Mel spectrogram input
 *
 * @param engine Engine instance
 * @param mel_input Contiguous Mel spectrogram [num_frames * 80]
 * @param num_frames Number of time frames
 * @param out_embedding Output L2-normalized speaker embedding (192-D or 512-D)
 */
esp_err_t sr_engine_infer(
    sr_engine_t *engine,
    const float *mel_input,
    size_t num_frames,
    float *out_embedding
);

#ifdef __cplusplus
}
#endif
