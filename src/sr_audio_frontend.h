#pragma once

#include "esp_err.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SR_MEL_BINS        80
#define SR_FRAME_LEN       400   // 25ms @ 16kHz
#define SR_FRAME_HOP       160   // 10ms @ 16kHz
#define SR_FFT_SIZE        512

/**
 * @brief Initialize audio frontend FFT tables
 */
esp_err_t sr_audio_frontend_init(void);

/**
 * @brief Deinitialize audio frontend
 */
void sr_audio_frontend_deinit(void);

/**
 * @brief Calculate number of frames for a given number of audio samples
 */
static inline size_t sr_audio_frontend_get_num_frames(size_t num_samples) {
    if (num_samples < SR_FRAME_LEN) {
        return 0;
    }
    return (num_samples - SR_FRAME_LEN) / SR_FRAME_HOP + 1;
}

/**
 * @brief Extract 80-bin Mel-spectrogram with CMN from 16kHz PCM audio
 *
 * @param pcm_samples Input 16kHz mono audio samples
 * @param num_samples Number of samples
 * @param out_mel Output buffer of size (80 * num_frames * sizeof(float)).
 *                Layout: [F=80, T=num_frames], contiguous in time for each frequency bin.
 * @param out_num_frames Output pointer to number of extracted frames
 * @return ESP_OK on success
 */
esp_err_t sr_audio_frontend_extract(
    const int16_t *pcm_samples,
    size_t num_samples,
    float *out_mel,
    size_t *out_num_frames
);

#ifdef __cplusplus
}
#endif
