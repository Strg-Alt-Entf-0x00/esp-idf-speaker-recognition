#ifndef AUDIO_WAV_H
#define AUDIO_WAV_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t sample_rate;
    uint16_t num_channels;
    uint16_t bits_per_sample;
    uint32_t data_size;
    uint32_t duration_ms;
} audio_wav_info_t;

typedef struct audio_wav_s* audio_wav_handle_t;

esp_err_t audio_wav_open(const char* filepath, audio_wav_handle_t* handle);
esp_err_t audio_wav_get_info(audio_wav_handle_t handle, audio_wav_info_t* info);
esp_err_t audio_wav_read(
    audio_wav_handle_t handle,
    void* buffer,
    size_t size,
    size_t* bytes_read);
esp_err_t audio_wav_seek(audio_wav_handle_t handle, size_t offset);
esp_err_t audio_wav_reset(audio_wav_handle_t handle);
esp_err_t audio_wav_close(audio_wav_handle_t handle);

esp_err_t audio_wav_load_file(
    const char* filepath,
    int16_t** buffer,
    size_t* buffer_size,
    audio_wav_info_t* info);

esp_err_t audio_wav_write_pcm16_header(
    FILE* file,
    uint32_t sample_rate,
    uint16_t num_channels,
    uint32_t data_size);

#ifdef __cplusplus
}
#endif

#endif
