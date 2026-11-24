/**
 * @file audio_wav.c
 * @brief PCM16 WAV file reader and writer
 */

#include "audio_wav.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

static const char* TAG = "AUDIO_WAV";

// WAV file format structures (little-endian)
#pragma pack(push, 1)
typedef struct {
    char riff_header[4];      // "RIFF"
    uint32_t wav_size;        // File size - 8
    char wave_header[4];      // "WAVE"
} wav_riff_header_t;

typedef struct {
    uint16_t audio_format;    // 1 = PCM
    uint16_t num_channels;    // 1 = mono, 2 = stereo
    uint32_t sample_rate;     // Sample rate (Hz)
    uint32_t byte_rate;       // sample_rate * num_channels * bits_per_sample / 8
    uint16_t block_align;     // num_channels * bits_per_sample / 8
    uint16_t bits_per_sample; // This component currently accepts PCM16 only.
} wav_fmt_chunk_t;

typedef struct {
    char id[4];
    uint32_t size;
} wav_chunk_header_t;
#pragma pack(pop)

struct audio_wav_s {
    FILE* file;
    audio_wav_info_t info;
    size_t data_start_offset;
    size_t current_offset;
};

/**
 * @brief Read and validate WAV header
 */
static esp_err_t read_wav_header(FILE* file, audio_wav_info_t* info, size_t* data_offset)
{
    wav_riff_header_t riff;
    if (fseek(file, 0, SEEK_END) != 0) {
        return ESP_ERR_INVALID_RESPONSE;
    }
    const long file_size = ftell(file);
    if (file_size < (long)sizeof(riff) || fseek(file, 0, SEEK_SET) != 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    if (fread(&riff, sizeof(riff), 1, file) != 1) {
        ESP_LOGE(TAG, "Failed to read RIFF header");
        return ESP_ERR_INVALID_ARG;
    }
    
    if (memcmp(riff.riff_header, "RIFF", 4) != 0) {
        ESP_LOGE(TAG, "Invalid RIFF header");
        return ESP_ERR_INVALID_ARG;
    }
    
    if (memcmp(riff.wave_header, "WAVE", 4) != 0) {
        ESP_LOGE(TAG, "Invalid WAVE header");
        return ESP_ERR_INVALID_ARG;
    }

    const uint64_t riff_end = (uint64_t)riff.wav_size + sizeof(riff.riff_header) +
        sizeof(riff.wav_size);
    if (riff.wav_size < sizeof(riff.wave_header) ||
        riff_end > (uint64_t)file_size || riff_end > LONG_MAX) {
        ESP_LOGE(TAG, "RIFF size exceeds the file or supported seek range");
        return ESP_ERR_INVALID_SIZE;
    }

    bool fmt_found = false;
    bool data_found = false;
    wav_fmt_chunk_t fmt = {0};
    uint32_t data_size = 0;
    size_t audio_data_offset = 0;

    while (true) {
        const long chunk_position = ftell(file);
        if (chunk_position < 0) {
            return ESP_ERR_INVALID_RESPONSE;
        }
        if ((uint64_t)chunk_position + sizeof(wav_chunk_header_t) > riff_end) {
            break;
        }
        wav_chunk_header_t chunk = {0};
        if (fread(&chunk, sizeof(chunk), 1, file) != 1) {
            return ESP_ERR_INVALID_ARG;
        }
        const long payload_position = ftell(file);
        if (payload_position < 0) {
            return ESP_ERR_INVALID_RESPONSE;
        }
        const uint64_t chunk_end = (uint64_t)payload_position + chunk.size;
        const uint64_t padded_end = chunk_end + (chunk.size & 1U);
        if (padded_end > riff_end || padded_end > LONG_MAX) {
            ESP_LOGE(TAG, "WAV chunk exceeds the RIFF boundary");
            return ESP_ERR_INVALID_SIZE;
        }

        if (memcmp(chunk.id, "fmt ", 4) == 0) {
            if (fmt_found || chunk.size < sizeof(fmt)) {
                ESP_LOGE(TAG, "Invalid or duplicate WAV format chunk");
                return ESP_ERR_INVALID_ARG;
            }
            if (fread(&fmt, sizeof(fmt), 1, file) != 1) {
                ESP_LOGE(TAG, "Failed to read WAV format data");
                return ESP_ERR_INVALID_ARG;
            }
            fmt_found = true;
        } else if (memcmp(chunk.id, "data", 4) == 0 && !data_found) {
            data_size = chunk.size;
            audio_data_offset = (size_t)payload_position;
            data_found = true;
        } else {
            ESP_LOGD(TAG, "Skipping WAV chunk: %.4s (%u bytes)",
                     chunk.id, (unsigned)chunk.size);
        }

        if (fseek(file, (long)padded_end, SEEK_SET) != 0) {
            ESP_LOGE(TAG, "Could not seek to the next WAV chunk");
            return ESP_ERR_INVALID_RESPONSE;
        }
        if (fmt_found && data_found) {
            break;
        }
    }

    if (!fmt_found || !data_found) {
        ESP_LOGE(TAG, "WAV file is missing a format or data chunk");
        return ESP_ERR_INVALID_ARG;
    }
    if (fmt.audio_format != 1) {
        ESP_LOGE(TAG, "Unsupported audio format: %u (only PCM supported)",
                 (unsigned)fmt.audio_format);
        return ESP_ERR_NOT_SUPPORTED;
    }
    if (fmt.sample_rate == 0 ||
        (fmt.num_channels != 1 && fmt.num_channels != 2) ||
        fmt.bits_per_sample != 16) {
        ESP_LOGE(TAG, "Unsupported WAV format: %u Hz, %u channels, %u-bit PCM; "
                      "expected non-zero sample rate, mono/stereo, 16-bit PCM",
                 (unsigned)fmt.sample_rate, (unsigned)fmt.num_channels,
                 (unsigned)fmt.bits_per_sample);
        return ESP_ERR_NOT_SUPPORTED;
    }
    const uint16_t expected_block_align =
        (uint16_t)(fmt.num_channels * fmt.bits_per_sample / 8U);
    const uint64_t expected_byte_rate =
        (uint64_t)fmt.sample_rate * expected_block_align;
    if (fmt.block_align != expected_block_align ||
        expected_byte_rate > UINT32_MAX || fmt.byte_rate != expected_byte_rate ||
        data_size == 0 || data_size % expected_block_align != 0) {
        ESP_LOGE(TAG, "Inconsistent WAV sample layout or empty audio data");
        return ESP_ERR_INVALID_SIZE;
    }

    info->sample_rate = fmt.sample_rate;
    info->num_channels = fmt.num_channels;
    info->bits_per_sample = fmt.bits_per_sample;
    info->data_size = data_size;
    const uint64_t samples_per_channel = data_size / expected_block_align;
    const uint64_t duration_ms =
        (samples_per_channel * 1000U) / fmt.sample_rate;
    if (duration_ms > UINT32_MAX) {
        ESP_LOGE(TAG, "WAV duration exceeds the supported range");
        return ESP_ERR_INVALID_SIZE;
    }

    info->duration_ms = (uint32_t)duration_ms;
    *data_offset = audio_data_offset;

    if (fseek(file, (long)audio_data_offset, SEEK_SET) != 0) {
        return ESP_ERR_INVALID_RESPONSE;
    }
    
    ESP_LOGI(TAG, "WAV file info:");
    ESP_LOGI(TAG, "  Sample rate: %u Hz", info->sample_rate);
    ESP_LOGI(TAG, "  Channels: %d", info->num_channels);
    ESP_LOGI(TAG, "  Bits/sample: %d", info->bits_per_sample);
    ESP_LOGI(TAG, "  Data size: %u bytes", info->data_size);
    ESP_LOGI(TAG, "  Duration: %u ms", info->duration_ms);
    
    return ESP_OK;
}

esp_err_t audio_wav_open(const char* filepath, audio_wav_handle_t* handle)
{
    if (filepath == NULL || handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    *handle = NULL;
    
    ESP_LOGI(TAG, "Opening WAV file: %s", filepath);
    
    // Allocate handle
    audio_wav_handle_t h = (audio_wav_handle_t)malloc(sizeof(struct audio_wav_s));
    if (h == NULL) {
        ESP_LOGE(TAG, "Failed to allocate handle");
        return ESP_ERR_NO_MEM;
    }
    memset(h, 0, sizeof(struct audio_wav_s));
    
    // Open file
    h->file = fopen(filepath, "rb");
    if (h->file == NULL) {
        ESP_LOGE(TAG, "Failed to open file: %s", filepath);
        free(h);
        return ESP_ERR_NOT_FOUND;
    }
    
    // Read and validate header
    esp_err_t ret = read_wav_header(h->file, &h->info, &h->data_start_offset);
    if (ret != ESP_OK) {
        fclose(h->file);
        free(h);
        return ret;
    }
    
    h->current_offset = 0;
    *handle = h;
    
    ESP_LOGI(TAG, "WAV file opened successfully");
    return ESP_OK;
}

esp_err_t audio_wav_get_info(audio_wav_handle_t handle, audio_wav_info_t* info)
{
    if (handle == NULL || info == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    memcpy(info, &handle->info, sizeof(audio_wav_info_t));
    return ESP_OK;
}

esp_err_t audio_wav_read(
    audio_wav_handle_t handle,
    void* buffer,
    size_t size,
    size_t* bytes_read)
{
    if (handle == NULL || buffer == NULL || bytes_read == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    *bytes_read = 0;
    
    // Calculate how much we can read
    if (handle->current_offset > handle->info.data_size) {
        return ESP_ERR_INVALID_STATE;
    }
    size_t remaining = handle->info.data_size - handle->current_offset;
    size_t to_read = (size < remaining) ? size : remaining;
    
    if (to_read == 0) {
        return ESP_OK; // EOF
    }
    
    // Read data
    size_t read = fread(buffer, 1, to_read, handle->file);
    if (read != to_read) {
        ESP_LOGE(TAG, "Failed to read data: expected %zu, got %zu", to_read, read);
        return ESP_ERR_INVALID_RESPONSE;
    }
    
    handle->current_offset += read;
    *bytes_read = read;
    
    return ESP_OK;
}

esp_err_t audio_wav_seek(audio_wav_handle_t handle, size_t offset)
{
    if (handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (offset > handle->info.data_size) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (offset > SIZE_MAX - handle->data_start_offset ||
        handle->data_start_offset + offset > LONG_MAX ||
        fseek(handle->file, (long)(handle->data_start_offset + offset), SEEK_SET) != 0) {
        ESP_LOGE(TAG, "Failed to seek to offset %zu", offset);
        return ESP_ERR_INVALID_RESPONSE;
    }
    
    handle->current_offset = offset;
    return ESP_OK;
}

esp_err_t audio_wav_reset(audio_wav_handle_t handle)
{
    return audio_wav_seek(handle, 0);
}

esp_err_t audio_wav_close(audio_wav_handle_t handle)
{
    if (handle == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    
    esp_err_t ret = ESP_OK;
    if (handle->file != NULL && fclose(handle->file) != 0) {
        ret = ESP_FAIL;
    }

    free(handle);
    ESP_LOGI(TAG, "WAV file closed");
    return ret;
}

esp_err_t audio_wav_load_file(
    const char* filepath,
    int16_t** buffer,
    size_t* buffer_size,
    audio_wav_info_t* info)
{
    if (filepath == NULL || buffer == NULL || buffer_size == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    *buffer = NULL;
    *buffer_size = 0;
    
    // Open file
    audio_wav_handle_t handle;
    esp_err_t ret = audio_wav_open(filepath, &handle);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Get info
    audio_wav_info_t file_info;
    ret = audio_wav_get_info(handle, &file_info);
    if (ret != ESP_OK) {
        audio_wav_close(handle);
        return ret;
    }
    
    // Allocate buffer in PSRAM if available
    int16_t* buf = (int16_t*)heap_caps_malloc(file_info.data_size, 
                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (buf == NULL) {
        // Try regular heap
        buf = (int16_t*)malloc(file_info.data_size);
        if (buf == NULL) {
            ESP_LOGE(TAG, "Failed to allocate buffer (%u bytes)", file_info.data_size);
            audio_wav_close(handle);
            return ESP_ERR_NO_MEM;
        }
    }
    
    // Read entire file
    size_t bytes_read;
    ret = audio_wav_read(handle, buf, file_info.data_size, &bytes_read);
    if (ret != ESP_OK || bytes_read != file_info.data_size) {
        ESP_LOGE(TAG, "Failed to read file data");
        free(buf);
        audio_wav_close(handle);
        return ESP_ERR_INVALID_RESPONSE;
    }
    
    // Close file
    ret = audio_wav_close(handle);
    if (ret != ESP_OK) {
        free(buf);
        return ret;
    }
    
    // Return results
    *buffer = buf;
    *buffer_size = file_info.data_size;
    if (info != NULL) {
        memcpy(info, &file_info, sizeof(audio_wav_info_t));
    }
    
    ESP_LOGI(TAG, "Loaded WAV file: %u bytes (%.2f seconds)", 
             file_info.data_size,
             (float)file_info.duration_ms / 1000.0f);
    
    return ESP_OK;
}

static void write_le16(uint8_t* destination, uint16_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
}

static void write_le32(uint8_t* destination, uint32_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8);
    destination[2] = (uint8_t)(value >> 16);
    destination[3] = (uint8_t)(value >> 24);
}

esp_err_t audio_wav_write_pcm16_header(
    FILE* file,
    uint32_t sample_rate,
    uint16_t num_channels,
    uint32_t data_size)
{
    if (file == NULL || sample_rate == 0 ||
        (num_channels != 1 && num_channels != 2) ||
        data_size > UINT32_MAX - 36U ||
        (data_size != 0 &&
         data_size % ((uint32_t)num_channels * sizeof(int16_t)) != 0)) {
        return ESP_ERR_INVALID_ARG;
    }

    const uint32_t block_align = (uint32_t)num_channels * sizeof(int16_t);
    const uint64_t byte_rate = (uint64_t)sample_rate * block_align;
    if (byte_rate > UINT32_MAX) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t header[44] = {0};
    memcpy(&header[0], "RIFF", 4);
    write_le32(&header[4], 36U + data_size);
    memcpy(&header[8], "WAVEfmt ", 8);
    write_le32(&header[16], 16);
    write_le16(&header[20], 1);
    write_le16(&header[22], num_channels);
    write_le32(&header[24], sample_rate);
    write_le32(&header[28], (uint32_t)byte_rate);
    write_le16(&header[32], (uint16_t)block_align);
    write_le16(&header[34], 16);
    memcpy(&header[36], "data", 4);
    write_le32(&header[40], data_size);

    if (fseek(file, 0, SEEK_SET) != 0 ||
        fwrite(header, 1, sizeof(header), file) != sizeof(header) ||
        fflush(file) != 0) {
        return ESP_FAIL;
    }
    return ESP_OK;
}
