#include "esp_speaker_recognition.h"
#include "sr_audio_frontend.h"
#include "sr_engine.h"
#include "sr_model.h"
#include "sr_simd_pie.h"
#include "sr_parallel.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TAG = "SR_DB";
static const char *NVS_NAMESPACE = "sr_db";

typedef struct {
    char name[ESP_SR_NAME_MAX_LEN];
    float embedding[ESP_SR_MAX_EMBEDDING_DIM];
    uint16_t dim;
    bool active;
} sr_profile_t;

static sr_model_t s_model;
static sr_engine_t *s_engine = NULL;
static SemaphoreHandle_t s_mutex = NULL;
static bool s_initialized = false;
static float s_verify_threshold = 0.30f;
static uint8_t *s_allocated_model_buffer = NULL;
static void *s_allocated_model_raw = NULL;

static sr_profile_t s_profiles[ESP_SR_MAX_SPEAKERS];
static size_t s_profile_count = 0;

static esp_err_t load_profiles_from_nvs(void) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &nvs);
    if (err != ESP_OK) {
        return err; // No NVS database yet
    }

    uint32_t count = 0;
    nvs_get_u32(nvs, "spk_count", &count);
    if (count > ESP_SR_MAX_SPEAKERS) {
        count = ESP_SR_MAX_SPEAKERS;
    }

    s_profile_count = 0;
    for (uint32_t i = 0; i < count; i++) {
        char key_name[16];
        char key_emb[16];
        snprintf(key_name, sizeof(key_name), "name_%lu", (unsigned long)i);
        snprintf(key_emb, sizeof(key_emb), "emb_%lu", (unsigned long)i);

        size_t name_len = ESP_SR_NAME_MAX_LEN;
        size_t emb_len = sizeof(s_profiles[i].embedding);

        if (nvs_get_str(nvs, key_name, s_profiles[i].name, &name_len) == ESP_OK &&
            nvs_get_blob(nvs, key_emb, s_profiles[i].embedding, &emb_len) == ESP_OK) {
            s_profiles[i].dim = (uint16_t)(emb_len / sizeof(float));
            s_profiles[i].active = true;
            s_profile_count++;
            ESP_LOGI(TAG, "Loaded enrolled speaker [%lu]: %s (dim=%u)", (unsigned long)i, s_profiles[i].name, (unsigned)s_profiles[i].dim);
        }
    }

    nvs_close(nvs);
    return ESP_OK;
}

static esp_err_t save_profiles_to_nvs(void) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs);
    if (err != ESP_OK) {
        return err;
    }

    nvs_set_u32(nvs, "spk_count", (uint32_t)s_profile_count);

    size_t idx = 0;
    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
        if (s_profiles[i].active) {
            char key_name[16];
            char key_emb[16];
            snprintf(key_name, sizeof(key_name), "name_%lu", (unsigned long)idx);
            snprintf(key_emb, sizeof(key_emb), "emb_%lu", (unsigned long)idx);

            size_t bytes = (s_profiles[i].dim > 0) ? (s_profiles[i].dim * sizeof(float))
                                                   : (esp_speaker_recognition_get_embedding_dim() * sizeof(float));

            nvs_set_str(nvs, key_name, s_profiles[i].name);
            nvs_set_blob(nvs, key_emb, s_profiles[i].embedding, bytes);
            idx++;
        }
    }

    nvs_commit(nvs);
    nvs_close(nvs);
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_init(const esp_sr_config_t *config) {
    if (!config || !config->model_data || config->model_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    if (s_mutex == NULL) {
        s_mutex = xSemaphoreCreateMutex();
        if (!s_mutex) return ESP_ERR_NO_MEM;
    }

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    if (s_initialized) {
        xSemaphoreGive(s_mutex);
        return ESP_OK;
    }

    s_verify_threshold = (config->verify_threshold > 0.0f) ? config->verify_threshold : 0.70f;

    // 1. Initialize Audio Frontend
    esp_err_t err = sr_audio_frontend_init();
    if (err != ESP_OK) {
        xSemaphoreGive(s_mutex);
        return err;
    }

    // 2. Load Model ESPM
    err = sr_model_load(config->model_data, config->model_size, &s_model);
    if (err != ESP_OK) {
        sr_audio_frontend_deinit();
        xSemaphoreGive(s_mutex);
        return err;
    }

    // 3. Create Inference Engine (default up to 320 frames = 3.2 seconds)
    err = sr_engine_create(&s_model, 320, config->use_psram, &s_engine);
    if (err != ESP_OK) {
        sr_audio_frontend_deinit();
        xSemaphoreGive(s_mutex);
        return err;
    }

    // 4. Load enrolled speakers from NVS
    memset(s_profiles, 0, sizeof(s_profiles));
    s_profile_count = 0;
    load_profiles_from_nvs();

    // 5. Initialize Dual-Core Worker
    sr_parallel_init();

    s_initialized = true;
    xSemaphoreGive(s_mutex);

    ESP_LOGI(TAG, "Speaker recognition system ready (Threshold=%.2f, Enrolled=%zu)",
             s_verify_threshold, s_profile_count);
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_init_partition(const char *partition_label, float verify_threshold) {
    const esp_partition_t *part = NULL;
    if (partition_label && partition_label[0] != '\0') {
        part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, partition_label);
    } else {
        part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x50, NULL);
        if (!part) {
            part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
        }
    }
    if (!part) {
        ESP_LOGE(TAG, "Speaker recognition model partition not found!");
        return ESP_ERR_NOT_FOUND;
    }

    ESP_LOGI(TAG, "Loading model from partition '%s' (size=%u KB)...",
             part->label, (unsigned)(part->size / 1024));

    void *raw = heap_caps_malloc(part->size + 32, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw) {
        raw = malloc(part->size + 32);
    }
    if (!raw) {
        ESP_LOGE(TAG, "Failed to allocate %u KB for model", (unsigned)(part->size / 1024));
        return ESP_ERR_NO_MEM;
    }
    uint8_t *buf = (uint8_t *)(((uintptr_t)raw + 15) & ~15);

    esp_err_t err = esp_partition_read(part, 0, buf, part->size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read model partition: %s", esp_err_to_name(err));
        free(raw);
        return err;
    }

    esp_sr_config_t cfg = {
        .model_data = buf,
        .model_size = part->size,
        .verify_threshold = verify_threshold,
        .use_psram = true,
    };
    err = esp_speaker_recognition_init(&cfg);
    if (err == ESP_OK) {
        s_allocated_model_buffer = buf;
        s_allocated_model_raw = raw;
    } else {
        free(raw);
    }
    return err;
}

esp_err_t esp_speaker_recognition_init_file(const char *file_path, float verify_threshold) {
    if (!file_path || file_path[0] == '\0') {
        return ESP_ERR_INVALID_ARG;
    }
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        ESP_LOGE(TAG, "Cannot open model file: %s", file_path);
        return ESP_ERR_NOT_FOUND;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return ESP_ERR_INVALID_SIZE;
    }

    ESP_LOGI(TAG, "Loading model from file '%s' (size=%ld KB)...", file_path, size / 1024);

    void *raw = heap_caps_malloc(size + 32, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!raw) {
        raw = malloc(size + 32);
    }
    if (!raw) {
        fclose(f);
        return ESP_ERR_NO_MEM;
    }
    uint8_t *buf = (uint8_t *)(((uintptr_t)raw + 15) & ~15);

    size_t read_bytes = fread(buf, 1, size, f);
    fclose(f);
    if (read_bytes != (size_t)size) {
        ESP_LOGE(TAG, "Failed to read complete model file (%zu of %ld bytes)", read_bytes, size);
        free(raw);
        return ESP_FAIL;
    }

    esp_sr_config_t cfg = {
        .model_data = buf,
        .model_size = (size_t)size,
        .verify_threshold = verify_threshold,
        .use_psram = true,
    };
    esp_err_t err = esp_speaker_recognition_init(&cfg);
    if (err == ESP_OK) {
        s_allocated_model_buffer = buf;
        s_allocated_model_raw = raw;
    } else {
        free(raw);
    }
    return err;
}

esp_err_t esp_speaker_recognition_deinit(void) {
    if (s_mutex == NULL || !s_initialized) {
        return ESP_OK;
    }

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    if (s_engine) {
        sr_engine_destroy(s_engine);
        s_engine = NULL;
    }
    sr_parallel_deinit();
    sr_audio_frontend_deinit();
    s_profile_count = 0;
    s_initialized = false;

    if (s_allocated_model_raw) {
        free(s_allocated_model_raw);
        s_allocated_model_raw = NULL;
        s_allocated_model_buffer = NULL;
    } else if (s_allocated_model_buffer) {
        free(s_allocated_model_buffer);
        s_allocated_model_buffer = NULL;
    }

    xSemaphoreGive(s_mutex);
    return ESP_OK;
}

bool esp_speaker_recognition_is_initialized(void) {
    return s_initialized;
}

float esp_speaker_recognition_get_threshold(void) {
    return s_verify_threshold;
}

esp_err_t esp_speaker_recognition_set_threshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        return ESP_ERR_INVALID_ARG;
    }
    s_verify_threshold = threshold;
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_extract_embedding(
    const int16_t *pcm_samples,
    size_t num_samples,
    float *out_embedding
) {
    if (!s_initialized || !s_engine) {
        return ESP_ERR_INVALID_STATE;
    }
    if (!pcm_samples || !out_embedding || num_samples < SR_FRAME_LEN) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    size_t num_frames = sr_audio_frontend_get_num_frames(num_samples);
    if (num_frames > 320) {
        num_frames = 320;
        num_samples = (num_frames - 1) * SR_FRAME_HOP + SR_FRAME_LEN;
    }

    // Allocate temporary mel buffer in PSRAM
    size_t mel_bytes = SR_MEL_BINS * num_frames * sizeof(float);
    float *mel_buf = (float *)heap_caps_malloc(mel_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!mel_buf) {
        mel_buf = (float *)malloc(mel_bytes);
    }
    if (!mel_buf) {
        xSemaphoreGive(s_mutex);
        return ESP_ERR_NO_MEM;
    }

    // 1. Audio frontend
    size_t actual_frames = 0;
    esp_err_t err = sr_audio_frontend_extract(pcm_samples, num_samples, mel_buf, &actual_frames);
    if (err != ESP_OK) {
        free(mel_buf);
        xSemaphoreGive(s_mutex);
        return err;
    }

    // 2. Engine inference
    err = sr_engine_infer(s_engine, mel_buf, actual_frames, out_embedding);
    free(mel_buf);

    xSemaphoreGive(s_mutex);
    return err;
}

int esp_speaker_recognition_get_embedding_dim(void) {
    if (s_engine) {
        return sr_engine_get_embedding_dim(s_engine);
    }
    return (s_model.embedding_dim > 0) ? s_model.embedding_dim : ESP_SR_EMBEDDING_DIM;
}

esp_sr_profile_t esp_speaker_recognition_get_profile(void) {
    if (s_model.arch == SR_ARCH_TDNN_TINY) {
        return ESP_SR_PROFILE_REALTIME;
    }
    return ESP_SR_PROFILE_HIGH_SECURITY;
}

esp_err_t esp_speaker_recognition_set_profile(esp_sr_profile_t profile) {
    const char *partition_label = (profile == ESP_SR_PROFILE_REALTIME) ? "model_rt" : "model";
    float default_threshold = (profile == ESP_SR_PROFILE_REALTIME) ? 0.985f : 0.30f;

    ESP_LOGI(TAG, "Switching profile to %s (partition: '%s')...",
             (profile == ESP_SR_PROFILE_REALTIME) ? "REALTIME (TDNN-Tiny)" : "HIGH_SECURITY (ECAPA-TDNN512)",
             partition_label);

    if (s_initialized) {
        esp_speaker_recognition_deinit();
    }

    esp_err_t err = esp_speaker_recognition_init_partition(partition_label, default_threshold);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to switch profile: partition '%s' error: %s", partition_label, esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "Profile switched successfully: %s (Dim=%d, Threshold=%.3f)",
             (profile == ESP_SR_PROFILE_REALTIME) ? "REALTIME" : "HIGH_SECURITY",
             esp_speaker_recognition_get_embedding_dim(),
             esp_speaker_recognition_get_threshold());
    return ESP_OK;
}

float esp_speaker_recognition_compute_similarity(
    const float *emb1,
    const float *emb2
) {
    int dim = esp_speaker_recognition_get_embedding_dim();
    return sr_simd_dot_product(emb1, emb2, dim);
}

esp_err_t esp_speaker_recognition_enroll_embedding(
    const char *speaker_name,
    const float *embedding
) {
    if (!speaker_name || !embedding || strlen(speaker_name) == 0) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    // Check if updating existing speaker
    int slot = -1;
    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
        if (s_profiles[i].active && strcmp(s_profiles[i].name, speaker_name) == 0) {
            slot = (int)i;
            break;
        }
    }

    if (slot == -1) {
        // Find empty slot
        for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
            if (!s_profiles[i].active) {
                slot = (int)i;
                break;
            }
        }
    }

    if (slot == -1) {
        xSemaphoreGive(s_mutex);
        return ESP_ERR_NO_MEM;
    }

    int dim = esp_speaker_recognition_get_embedding_dim();
    snprintf(s_profiles[slot].name, sizeof(s_profiles[slot].name), "%s", speaker_name);
    memcpy(s_profiles[slot].embedding, embedding, dim * sizeof(float));
    s_profiles[slot].dim = (uint16_t)dim;
    if (!s_profiles[slot].active) {
        s_profiles[slot].active = true;
        s_profile_count++;
    }

    save_profiles_to_nvs();
    xSemaphoreGive(s_mutex);

    ESP_LOGI(TAG, "Enrolled speaker '%s' at slot %d (dim=%d)", speaker_name, slot, dim);
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_enroll(
    const char *speaker_name,
    const int16_t *pcm_samples,
    size_t num_samples
) {
    float emb[ESP_SR_MAX_EMBEDDING_DIM];
    esp_err_t err = esp_speaker_recognition_extract_embedding(pcm_samples, num_samples, emb);
    if (err != ESP_OK) return err;

    return esp_speaker_recognition_enroll_embedding(speaker_name, emb);
}

esp_err_t esp_speaker_recognition_verify(
    const char *speaker_name,
    const int16_t *pcm_samples,
    size_t num_samples,
    esp_sr_verify_result_t *result
) {
    if (!speaker_name || !result) {
        return ESP_ERR_INVALID_ARG;
    }

    memset(result, 0, sizeof(*result));
    snprintf(result->matched_name, sizeof(result->matched_name), "%s", speaker_name);

    float probe_emb[ESP_SR_MAX_EMBEDDING_DIM];
    esp_err_t err = esp_speaker_recognition_extract_embedding(pcm_samples, num_samples, probe_emb);
    if (err != ESP_OK) return err;

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    int target_idx = -1;
    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
        if (s_profiles[i].active && strcmp(s_profiles[i].name, speaker_name) == 0) {
            target_idx = (int)i;
            break;
        }
    }

    if (target_idx == -1) {
        xSemaphoreGive(s_mutex);
        return ESP_ERR_NOT_FOUND;
    }

    int cur_dim = esp_speaker_recognition_get_embedding_dim();
    if (s_profiles[target_idx].dim != 0 && s_profiles[target_idx].dim != cur_dim) {
        ESP_LOGW(TAG, "Warning: Speaker '%s' enrolled with dim %u but current model is %d-D!",
                 speaker_name, (unsigned)s_profiles[target_idx].dim, cur_dim);
    }

    float sim = esp_speaker_recognition_compute_similarity(probe_emb, s_profiles[target_idx].embedding);
    result->similarity_score = sim;
    result->verified = (sim >= s_verify_threshold);

    xSemaphoreGive(s_mutex);
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_identify(
    const int16_t *pcm_samples,
    size_t num_samples,
    esp_sr_identify_result_t *result
) {
    if (!result) return ESP_ERR_INVALID_ARG;
    memset(result, 0, sizeof(*result));
    result->speaker_index = -1;

    float probe_emb[ESP_SR_MAX_EMBEDDING_DIM];
    esp_err_t err = esp_speaker_recognition_extract_embedding(pcm_samples, num_samples, probe_emb);
    if (err != ESP_OK) return err;

    xSemaphoreTake(s_mutex, portMAX_DELAY);

    float max_sim = -2.0f;
    int best_idx = -1;

    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
        if (s_profiles[i].active) {
            float sim = esp_speaker_recognition_compute_similarity(probe_emb, s_profiles[i].embedding);
            if (sim > max_sim) {
                max_sim = sim;
                best_idx = (int)i;
            }
        }
    }

    if (best_idx >= 0 && max_sim >= s_verify_threshold) {
        snprintf(result->matched_name, sizeof(result->matched_name), "%s", s_profiles[best_idx].name);
        result->similarity_score = max_sim;
        result->speaker_index = best_idx;
        result->identified = true;
    } else if (best_idx >= 0) {
        snprintf(result->matched_name, sizeof(result->matched_name), "%s", s_profiles[best_idx].name);
        result->similarity_score = max_sim;
        result->speaker_index = best_idx;
        result->identified = false;
    }

    xSemaphoreGive(s_mutex);
    return ESP_OK;
}

esp_err_t esp_speaker_recognition_delete(const char *speaker_name) {
    if (!speaker_name) return ESP_ERR_INVALID_ARG;

    xSemaphoreTake(s_mutex, portMAX_DELAY);
    bool found = false;

    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS; i++) {
        if (s_profiles[i].active && strcmp(s_profiles[i].name, speaker_name) == 0) {
            memset(&s_profiles[i], 0, sizeof(s_profiles[i]));
            s_profile_count--;
            found = true;
            break;
        }
    }

    if (found) {
        save_profiles_to_nvs();
    }
    xSemaphoreGive(s_mutex);

    return found ? ESP_OK : ESP_ERR_NOT_FOUND;
}

esp_err_t esp_speaker_recognition_clear_all(void) {
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    memset(s_profiles, 0, sizeof(s_profiles));
    s_profile_count = 0;

    nvs_handle_t nvs;
    if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &nvs) == ESP_OK) {
        nvs_erase_all(nvs);
        nvs_commit(nvs);
        nvs_close(nvs);
    }

    xSemaphoreGive(s_mutex);
    return ESP_OK;
}

size_t esp_speaker_recognition_get_count(void) {
    return s_profile_count;
}

size_t esp_speaker_recognition_get_names(char names[][ESP_SR_NAME_MAX_LEN], size_t max_count) {
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    size_t count = 0;
    for (size_t i = 0; i < ESP_SR_MAX_SPEAKERS && count < max_count; i++) {
        if (s_profiles[i].active) {
            size_t copy_len = strlen(s_profiles[i].name);
            if (copy_len >= ESP_SR_NAME_MAX_LEN) {
                copy_len = ESP_SR_NAME_MAX_LEN - 1;
            }
            memcpy(names[count], s_profiles[i].name, copy_len);
            names[count][copy_len] = '\0';
            count++;
        }
    }
    xSemaphoreGive(s_mutex);
    return count;
}
