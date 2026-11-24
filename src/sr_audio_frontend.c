#include "sr_audio_frontend.h"
#include "sr_mel_constants.h"
#include "dsps_fft2r.h"
#include "esp_log.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "SR_FRONTEND";

static bool s_initialized = false;
static float s_hamming_window[SR_FRAME_LEN];

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

esp_err_t sr_audio_frontend_init(void) {
    if (s_initialized) {
        return ESP_OK;
    }

    // Initialize 512-point FFT tables in ESP-DSP
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, SR_FFT_SIZE);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize ESP-DSP FFT tables: %d", ret);
        return ret;
    }

    // Precalculate Hamming window (Kaldi style: 0.54 - 0.46 * cos(2*pi*i / (N-1)))
    for (int i = 0; i < SR_FRAME_LEN; i++) {
        s_hamming_window[i] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (float)i / (float)(SR_FRAME_LEN - 1));
    }

    s_initialized = true;
    ESP_LOGI(TAG, "Audio frontend initialized (512-point FFT, 80-bin Mel)");
    return ESP_OK;
}

void sr_audio_frontend_deinit(void) {
    if (!s_initialized) {
        return;
    }
    dsps_fft2r_deinit_fc32();
    s_initialized = false;
}

esp_err_t sr_audio_frontend_extract(
    const int16_t *pcm_samples,
    size_t num_samples,
    float *out_mel,
    size_t *out_num_frames
) {
    if (!s_initialized) {
        esp_err_t err = sr_audio_frontend_init();
        if (err != ESP_OK) {
            return err;
        }
    }

    if (!pcm_samples || !out_mel || !out_num_frames) {
        return ESP_ERR_INVALID_ARG;
    }

    size_t num_frames = sr_audio_frontend_get_num_frames(num_samples);
    if (num_frames == 0) {
        *out_num_frames = 0;
        return ESP_ERR_INVALID_SIZE;
    }

    static float frame[SR_FRAME_LEN];
    static float fft_buf[SR_FFT_SIZE * 2]; // interleaved complex [re, im, re, im...]
    static float power_spectrum[SR_FFT_SIZE / 2 + 1];

    for (size_t t = 0; t < num_frames; t++) {
        const int16_t *frame_src = pcm_samples + t * SR_FRAME_HOP;

        // 1. Copy raw samples and compute mean for DC removal
        float dc_sum = 0.0f;
        for (int i = 0; i < SR_FRAME_LEN; i++) {
            frame[i] = (float)frame_src[i];
            dc_sum += frame[i];
        }
        float dc_mean = dc_sum / (float)SR_FRAME_LEN;

        // 2. Remove DC offset
        for (int i = 0; i < SR_FRAME_LEN; i++) {
            frame[i] -= dc_mean;
        }

        // 3. Pre-emphasis (0.97, Kaldi style)
        for (int i = SR_FRAME_LEN - 1; i > 0; i--) {
            frame[i] -= 0.97f * frame[i - 1];
        }
        frame[0] -= 0.97f * frame[0];

        // 4. Apply Hamming window and pack into FFT buffer
        for (int i = 0; i < SR_FRAME_LEN; i++) {
            fft_buf[2 * i] = frame[i] * s_hamming_window[i];
            fft_buf[2 * i + 1] = 0.0f;
        }
        // Zero-pad from 400 to 512
        for (int i = SR_FRAME_LEN; i < SR_FFT_SIZE; i++) {
            fft_buf[2 * i] = 0.0f;
            fft_buf[2 * i + 1] = 0.0f;
        }

        // 5. Compute FFT via ESP-DSP
        dsps_fft2r_fc32(fft_buf, SR_FFT_SIZE);
        dsps_bit_rev_fc32(fft_buf, SR_FFT_SIZE);

        // 6. Compute power spectrum (|X|^2)
        power_spectrum[0] = fft_buf[0] * fft_buf[0]; // DC
        for (int k = 1; k < SR_FFT_SIZE / 2; k++) {
            float re = fft_buf[2 * k];
            float im = fft_buf[2 * k + 1];
            power_spectrum[k] = re * re + im * im;
        }
        // Nyquist bin (k = 256)
        power_spectrum[SR_FFT_SIZE / 2] = fft_buf[1] * fft_buf[1];

        // 7. Apply 80 Mel triangular filters and log
        for (int m = 0; m < SR_MEL_BINS; m++) {
            uint8_t count = KALDI_MEL_COUNTS[m];
            uint16_t offset = KALDI_MEL_OFFSETS[m];
            float energy = 0.0f;

            for (int j = 0; j < count; j++) {
                uint16_t bin = KALDI_MEL_INDICES[offset + j];
                float weight = KALDI_MEL_WEIGHTS[offset + j];
                energy += power_spectrum[bin] * weight;
            }

            if (energy < 1e-6f) {
                energy = 1e-6f;
            }
            float log_energy = logf(energy);

            // Output format: [t * SR_MEL_BINS + m] (contiguous in frequency for each time frame)
            out_mel[t * SR_MEL_BINS + m] = log_energy;
        }
    }

    // 8. Cepstral Mean Normalization (CMN) along time dimension for each mel bin
    for (int m = 0; m < SR_MEL_BINS; m++) {
        float sum = 0.0f;
        for (size_t t = 0; t < num_frames; t++) {
            sum += out_mel[t * SR_MEL_BINS + m];
        }
        float mean = sum / (float)num_frames;
        for (size_t t = 0; t < num_frames; t++) {
            out_mel[t * SR_MEL_BINS + m] -= mean;
        }
    }

    *out_num_frames = num_frames;
    return ESP_OK;
}
