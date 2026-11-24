/**
 * @file audio_manager.h
 * @brief Speaker Recognition Audio Manager - Waveshare ESP32-P4-WIFI6 Board
 * 
 * Manages ES8311 codec (I2S0) and INMP441 microphone (I2S1) for audio
 * capture and playback. Provides ring-buffered audio pipeline for VAD processing.
 */

#ifndef AUDIO_MANAGER_H
#define AUDIO_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Microphone Types
// ============================================================================

typedef enum {
    MIC_I2S0_ES8311 = 0,    ///< ES8311 codec microphone (on-board)
    MIC_I2S1_INMP441 = 1    ///< INMP441 MEMS microphone (external)
} mic_type_t;

// ============================================================================
// Initialization & Status
// ============================================================================

/**
 * @brief Initialize audio manager (I2C, I2S, codec, ring buffers, tasks)
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t audio_manager_init(void);

/**
 * @brief Check if audio manager is initialized
 * 
 * @return true if initialized, false otherwise
 */
bool audio_manager_is_initialized(void);

// ============================================================================
// Microphone Configuration
// ============================================================================

/**
 * @brief Select and save the active microphone input
 * 
 * @param mic Microphone type (onboard ES8311 or external INMP441)
 * @return ESP_OK on success
 */
esp_err_t audio_manager_set_mic(mic_type_t mic);

/**
 * @brief Get currently active microphone
 * 
 * @return Current microphone type
 */
mic_type_t audio_manager_get_mic(void);

/**
 * @brief Set and save ES8311 microphone gain
 * 
 * @param gain_db Gain in dB (0, 6, 12, 18, 24, 30, 36, or 42)
 * @return ESP_OK on success
 */
esp_err_t audio_manager_set_mic_gain(float gain_db);

/**
 * @brief Get current ES8311 microphone gain
 * 
 * @param[out] gain_db Pointer to store gain value
 * @return ESP_OK on success
 */
esp_err_t audio_manager_get_mic_gain(float *gain_db);

// ============================================================================
// Speaker Configuration
// ============================================================================

/**
 * @brief Set ES8311 speaker volume
 * 
 * @param vol_level Volume level (0-100%)
 * @return ESP_OK on success
 */
esp_err_t audio_manager_set_speaker_vol(uint8_t vol_level);

// ============================================================================
// Audio Capture
// ============================================================================

/**
 * @brief Start audio capture from selected microphone
 * 
 * @return ESP_OK on success
 */
esp_err_t audio_manager_start_capture(void);

/**
 * @brief Stop audio capture
 * 
 * @return ESP_OK on success
 */
esp_err_t audio_manager_stop_capture(void);

/**
 * @brief Check if audio capture is running
 * 
 * @return true if capturing, false otherwise
 */
bool audio_manager_is_capturing(void);

/**
 * @brief Read captured audio samples (blocking)
 * 
 * @param[out] buffer Output buffer for audio samples
 * @param num_samples Number of samples to read
 * @param timeout_ms Timeout in milliseconds
 * @return Number of samples read, negative on error
 */
int audio_manager_read(int16_t *buffer, int num_samples, int timeout_ms);

// ============================================================================
// Audio Playback
// ============================================================================

/**
 * @brief Start audio playback
 * 
 * @return ESP_OK on success
 */
esp_err_t audio_manager_start_playback(void);

/**
 * @brief Stop audio playback
 * 
 * @return ESP_OK on success
 */
esp_err_t audio_manager_stop_playback(void);

/**
 * @brief Check if audio playback is running
 * 
 * @return true if playing, false otherwise
 */
bool audio_manager_is_playing(void);

/**
 * @brief Write audio samples for playback (blocking)
 * 
 * @param buffer Audio samples to play
 * @param num_samples Number of samples
 * @param timeout_ms Timeout in milliseconds
 * @return Number of samples written, negative on error
 */
int audio_manager_write(const int16_t *buffer, int num_samples, int timeout_ms);

// ============================================================================
// Utilities
// ============================================================================

/**
 * @brief Play boot chime sequence (4 ascending tones)
 * 
 * Plays a pleasant 4-note startup sequence to indicate system ready.
 */
void audio_manager_play_boot_sequence(void);

#ifdef __cplusplus
}
#endif

#endif // AUDIO_MANAGER_H
