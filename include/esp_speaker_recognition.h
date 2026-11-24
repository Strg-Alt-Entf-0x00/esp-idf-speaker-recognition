#pragma once

#include "esp_err.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESP_SR_MAX_EMBEDDING_DIM 512
#define ESP_SR_EMBEDDING_DIM     192
#define ESP_SR_NAME_MAX_LEN      32
#define ESP_SR_MAX_SPEAKERS      16

/**
 * @brief Speaker Recognition Model Profiles
 */
typedef enum {
    ESP_SR_PROFILE_HIGH_SECURITY = 0, /*!< ECAPA-TDNN512 (~2.6s, SOTA 0.8% EER, 192-D) */
    ESP_SR_PROFILE_REALTIME      = 1, /*!< TDNN-Tiny (~140ms, Sub-Second Voice, 512-D) */
} esp_sr_profile_t;

/**
 * @brief Result of 1:1 speaker verification
 */
typedef struct {
    char matched_name[ESP_SR_NAME_MAX_LEN];
    float similarity_score;
    bool verified;
} esp_sr_verify_result_t;

/**
 * @brief Result of 1:N speaker identification
 */
typedef struct {
    char matched_name[ESP_SR_NAME_MAX_LEN];
    float similarity_score;
    int speaker_index;
    bool identified;
} esp_sr_identify_result_t;

/**
 * @brief Configuration for speaker recognition engine
 */
typedef struct {
    const uint8_t *model_data;      /*!< Pointer to ESPM model in memory */
    size_t model_size;              /*!< Total size in bytes of ESPM model */
    float verify_threshold;         /*!< Verification threshold (e.g., 0.65 to 0.75) */
    bool use_psram;                 /*!< Allocate activation ping-pong buffers in PSRAM */
} esp_sr_config_t;

/**
 * @brief Initialize speaker recognition engine with an ESPM v3 model
 *
 * @param config Configuration pointer
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t esp_speaker_recognition_init(const esp_sr_config_t *config);

/**
 * @brief Convenience function to initialize speaker recognition directly from a Flash partition
 *
 * @param partition_label Partition name (e.g. "model"), or NULL to find partition with subtype 0x50
 * @param verify_threshold Verification threshold (e.g. 0.70f)
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_init_partition(const char *partition_label, float verify_threshold);

/**
 * @brief Convenience function to initialize speaker recognition directly from an ESPM file (e.g. on SD card or VFS)
 *
 * @param file_path Full filesystem path to the .espm model file
 * @param verify_threshold Verification threshold (e.g. 0.70f)
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_init_file(const char *file_path, float verify_threshold);

/**
 * @brief Deinitialize speaker recognition engine and free all buffers
 *
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_deinit(void);

/**
 * @brief Check if speaker recognition engine is initialized
 */
bool esp_speaker_recognition_is_initialized(void);

/**
 * @brief Get active speaker recognition profile
 */
esp_sr_profile_t esp_speaker_recognition_get_profile(void);

/**
 * @brief Switch active model profile (dynamically loads corresponding model from partition)
 *
 * @param profile ESP_SR_PROFILE_HIGH_SECURITY or ESP_SR_PROFILE_REALTIME
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_set_profile(esp_sr_profile_t profile);

/**
 * @brief Get output embedding dimension (e.g. 192 for ECAPA, 512 for TDNN-Tiny)
 */
int esp_speaker_recognition_get_embedding_dim(void);

/**
 * @brief Get current verification threshold
 */
float esp_speaker_recognition_get_threshold(void);

/**
 * @brief Set verification threshold (0.0f .. 1.0f)
 */
esp_err_t esp_speaker_recognition_set_threshold(float threshold);

/**
 * @brief Extract an L2-normalized speaker embedding from 16 kHz mono PCM audio
 *
 * @param pcm_samples Audio buffer (16 kHz mono signed 16-bit)
 * @param num_samples Number of samples (recommended: 1.5s to 3.0s, i.e. 24,000 to 48,000 samples)
 * @param out_embedding Output array of float elements (at least get_embedding_dim() elements)
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_extract_embedding(
    const int16_t *pcm_samples,
    size_t num_samples,
    float *out_embedding
);

/**
 * @brief Compute cosine similarity between two normalized embeddings
 *
 * @param emb1 First embedding
 * @param emb2 Second embedding
 * @return Dot product cosine similarity score [-1.0 .. +1.0]
 */
float esp_speaker_recognition_compute_similarity(
    const float *emb1,
    const float *emb2
);

/**
 * @brief Enroll a new speaker into the database from audio samples
 *
 * @param speaker_name Unique identifier for speaker (max 31 characters)
 * @param pcm_samples 16 kHz audio samples
 * @param num_samples Number of audio samples
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_enroll(
    const char *speaker_name,
    const int16_t *pcm_samples,
    size_t num_samples
);

/**
 * @brief Enroll a speaker profile directly from an embedding
 *
 * @param speaker_name Speaker name
 * @param embedding Precomputed normalized embedding
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_enroll_embedding(
    const char *speaker_name,
    const float *embedding
);

/**
 * @brief Verify a speaker (1:1 matching) against an enrolled identity
 *
 * @param speaker_name Expected speaker name
 * @param pcm_samples Audio samples
 * @param num_samples Number of samples
 * @param result Output verification details
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_verify(
    const char *speaker_name,
    const int16_t *pcm_samples,
    size_t num_samples,
    esp_sr_verify_result_t *result
);

/**
 * @brief Identify a speaker (1:N matching) against all enrolled profiles
 *
 * @param pcm_samples Audio samples
 * @param num_samples Number of samples
 * @param result Output identification details
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_identify(
    const int16_t *pcm_samples,
    size_t num_samples,
    esp_sr_identify_result_t *result
);

/**
 * @brief Delete an enrolled speaker from memory and NVS
 *
 * @param speaker_name Name of speaker to remove
 * @return ESP_OK on success
 */
esp_err_t esp_speaker_recognition_delete(const char *speaker_name);

/**
 * @brief Delete all enrolled speakers from memory and NVS
 */
esp_err_t esp_speaker_recognition_clear_all(void);

/**
 * @brief Get total number of enrolled speakers
 */
size_t esp_speaker_recognition_get_count(void);

/**
 * @brief Get names of all currently enrolled speakers
 *
 * @param names Output array of names
 * @param max_count Maximum names to copy
 * @return Number of names retrieved
 */
size_t esp_speaker_recognition_get_names(char names[][ESP_SR_NAME_MAX_LEN], size_t max_count);

/**
 * @brief Hardware-accelerated 1D INT8 Dot Product via ESP32-P4 RV32 PIE SIMD
 * Computes 16 signed 8-bit MACs per cycle using hardware accumulator.
 */
esp_err_t sr_dotprod_s8_pie(const int8_t *src1, const int8_t *src2, int32_t *dest, int len);

#ifdef __cplusplus
}
#endif
