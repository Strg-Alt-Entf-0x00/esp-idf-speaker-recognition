/**
 * @file audio_manager.cpp
 * @brief Speaker Recognition Audio Manager Implementation (Developer Edition)
 * 
 * Industrial-grade multi-threaded audio pipeline for Waveshare ESP32-P4-WIFI6:
 * - ES8311 Codec (I2S0): Primary audio interface with full duplex
 * - INMP441 MEMS Mic (I2S1): Secondary microphone (digital PDM)
 * - Ring-buffered capture and playback with zero-copy DMA
 * - Fault-tolerant: Works with or without codec hardware
 * 
 * Based on proven FireRedVAD audio architecture
 */

#include "audio_manager.h"
#include "board_audio_config.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "esp_err.h"
#include "nvs.h"
#include "driver/i2c_master.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "es8311_codec.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"

static const char* TAG = "audio_mgr";

// ============================================================================
// Global State
// ============================================================================

static bool g_initialized = false;
static volatile bool g_capture_running = false;
static volatile bool g_playback_running = false;
static float g_current_mic_gain = (float)BOARD_AUDIO_DEFAULT_MIC_GAIN_DB;
static uint8_t g_current_speaker_vol = BOARD_AUDIO_DEFAULT_SPEAKER_VOLUME;
static mic_type_t g_active_mic = MIC_I2S0_ES8311;

static const char *AUDIO_SETTINGS_NAMESPACE = "audio_cfg";
static const char *AUDIO_SETTING_SPEAKER_VOLUME = "speaker_vol";
static const char *AUDIO_SETTING_MIC_GAIN = "mic_gain";
static const char *AUDIO_SETTING_MIC_INPUT = "mic_input";

static bool is_valid_mic_gain(uint8_t gain_db) {
    return gain_db <= 42 && gain_db % 6 == 0;
}

// Ring Buffers
static RingbufHandle_t g_tx_ringbuf = NULL;
static RingbufHandle_t g_rx_ringbuf = NULL;

// Task Handles
static TaskHandle_t g_tx_task_handle = NULL;
static TaskHandle_t g_rx_task_handle = NULL;

// Hardware Handles
static i2c_master_bus_handle_t g_i2c_bus_handle = NULL;
static i2s_chan_handle_t g_i2s_rx_handle = NULL;
static i2s_chan_handle_t g_i2s_tx_handle = NULL;
static i2s_chan_handle_t g_i2s1_rx_handle = NULL;

// Codec Interfaces (full separation for professional codec control)
static const audio_codec_ctrl_if_t *g_codec_ctrl_if = NULL;
static const audio_codec_data_if_t *g_codec_data_if = NULL;
static const audio_codec_gpio_if_t *g_codec_gpio_if = NULL;
static const audio_codec_if_t *g_codec_if = NULL;

static esp_err_t apply_speaker_volume(uint8_t vol_level) {
    if (vol_level > 100) return ESP_ERR_INVALID_ARG;

    const float db_val = vol_level == 0
        ? -90.0f
        : 20.0f * log10f((float)vol_level / 100.0f);
    return g_codec_if->set_vol(g_codec_if, db_val);
}

static void load_audio_settings(void) {
    nvs_handle_t handle;
    esp_err_t ret = nvs_open(AUDIO_SETTINGS_NAMESPACE, NVS_READONLY, &handle);
    if (ret == ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGI(TAG, "No saved speaker volume; using default %u%%",
                 g_current_speaker_vol);
        return;
    }
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Could not open saved audio settings; using defaults: %s",
                 esp_err_to_name(ret));
        return;
    }

    uint8_t value = 0;
    ret = nvs_get_u8(handle, AUDIO_SETTING_SPEAKER_VOLUME, &value);
    if (ret == ESP_OK) {
        if (value <= 100) {
            g_current_speaker_vol = value;
            ESP_LOGI(TAG, "Restored speaker volume: %u%%", value);
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved speaker volume %u", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved speaker volume: %s",
                 esp_err_to_name(ret));
    }

    ret = nvs_get_u8(handle, AUDIO_SETTING_MIC_GAIN, &value);
    if (ret == ESP_OK) {
        if (is_valid_mic_gain(value)) {
            g_current_mic_gain = (float)value;
            ESP_LOGI(TAG, "Restored onboard ES8311 microphone gain: %u dB", value);
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved microphone gain %u dB", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved microphone gain: %s",
                 esp_err_to_name(ret));
    }

    ret = nvs_get_u8(handle, AUDIO_SETTING_MIC_INPUT, &value);
    if (ret == ESP_OK) {
        if (value == 0) {
            g_active_mic = MIC_I2S0_ES8311;
            ESP_LOGI(TAG, "Restored onboard ES8311 microphone");
        } else if (value == 1) {
            g_active_mic = MIC_I2S1_INMP441;
            ESP_LOGI(TAG, "Restored external INMP441 microphone");
        } else {
            ESP_LOGW(TAG, "Ignoring invalid saved microphone input %u", value);
        }
    } else if (ret != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "Could not read saved microphone input: %s",
                 esp_err_to_name(ret));
    }

    nvs_close(handle);
}

static esp_err_t save_audio_setting(const char *key, uint8_t value) {
    nvs_handle_t handle;
    esp_err_t ret = nvs_open(AUDIO_SETTINGS_NAMESPACE, NVS_READWRITE, &handle);
    if (ret != ESP_OK) return ret;

    ret = nvs_set_u8(handle, key, value);
    if (ret == ESP_OK) ret = nvs_commit(handle);
    nvs_close(handle);
    return ret;
}

// ============================================================================
// I2C Bus Initialization
// ============================================================================

static esp_err_t audio_i2c_init(void) {
    i2c_master_bus_config_t i2c_mst_config = {};
    i2c_mst_config.clk_source = I2C_CLK_SRC_DEFAULT;
    i2c_mst_config.i2c_port = I2C_NUM_0;
    i2c_mst_config.scl_io_num = (gpio_num_t)BOARD_AUDIO_I2C_SCL_GPIO;
    i2c_mst_config.sda_io_num = (gpio_num_t)BOARD_AUDIO_I2C_SDA_GPIO;
    i2c_mst_config.glitch_ignore_cnt = 7;
    i2c_mst_config.flags.enable_internal_pullup = true;
    
    ESP_LOGI(TAG, "I2C: SCL=%d, SDA=%d", BOARD_AUDIO_I2C_SCL_GPIO, BOARD_AUDIO_I2C_SDA_GPIO);
    
    return i2c_new_master_bus(&i2c_mst_config, &g_i2c_bus_handle);
}

// ============================================================================
// I2S0 (ES8311) Initialization - Full Duplex
// ============================================================================

static esp_err_t audio_i2s0_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    esp_err_t ret = i2s_new_channel(&chan_cfg, &g_i2s_tx_handle, &g_i2s_rx_handle);
    if (ret != ESP_OK) return ret;
    
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(BOARD_AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            (i2s_data_bit_width_t)BOARD_AUDIO_BITS_PER_SAMPLE,
            I2S_SLOT_MODE_STEREO
        ),
        .gpio_cfg = {
            .mclk = (gpio_num_t)BOARD_AUDIO_I2S_MCLK_GPIO,
            .bclk = (gpio_num_t)BOARD_AUDIO_I2S_BCLK_GPIO,
            .ws   = (gpio_num_t)BOARD_AUDIO_I2S_WS_GPIO,
            .dout = (gpio_num_t)BOARD_AUDIO_I2S_DOUT_GPIO,
            .din  = (gpio_num_t)BOARD_AUDIO_I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_APLL;
    
    ret = i2s_channel_init_std_mode(g_i2s_rx_handle, &std_cfg);
    if (ret != ESP_OK) return ret;
    
    ret = i2s_channel_init_std_mode(g_i2s_tx_handle, &std_cfg);
    if (ret != ESP_OK) return ret;
    
    ESP_LOGI(TAG, "I2S0 (ES8311): RX+TX initialized @ %dHz", BOARD_AUDIO_SAMPLE_RATE);
    return ESP_OK;
}

// ============================================================================
// I2S1 (INMP441) Initialization - Digital PDM Microphone
// ============================================================================

static esp_err_t audio_i2s1_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;
    
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &g_i2s1_rx_handle);
    if (ret != ESP_OK) return ret;
    
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(BOARD_AUDIO_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
            (i2s_data_bit_width_t)BOARD_AUDIO_BITS_PER_SAMPLE,
            I2S_SLOT_MODE_STEREO
        ),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)BOARD_AUDIO_INMP441_BCLK_GPIO,
            .ws   = (gpio_num_t)BOARD_AUDIO_INMP441_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)BOARD_AUDIO_INMP441_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.clk_cfg.clk_src = I2S_CLK_SRC_APLL;
    
    ret = i2s_channel_init_std_mode(g_i2s1_rx_handle, &std_cfg);
    if (ret != ESP_OK) return ret;
    
    // Enable I2S1 immediately (always-on for fast switching)
    ret = i2s_channel_enable(g_i2s1_rx_handle);
    
    ESP_LOGI(TAG, "I2S1 (INMP441): RX initialized @ %dHz", BOARD_AUDIO_SAMPLE_RATE);
    return ret;
}

// ============================================================================
// ES8311 Codec Initialization - Professional Grade
// ============================================================================

static esp_err_t audio_codec_init(void) {
    // Step 1: Create I2C Control Interface
    audio_codec_i2c_cfg_t i2c_cfg = {};
    i2c_cfg.port = I2C_NUM_0;
    i2c_cfg.addr = ES8311_CODEC_DEFAULT_ADDR;
    i2c_cfg.bus_handle = g_i2c_bus_handle;
    
    g_codec_ctrl_if = audio_codec_new_i2c_ctrl(&i2c_cfg);
    if (!g_codec_ctrl_if) {
        ESP_LOGW(TAG, "Failed to create I2C control interface");
        return ESP_FAIL;
    }
    
    // Step 2: Create I2S Data Interface
    audio_codec_i2s_cfg_t i2s_cfg = {};
    i2s_cfg.port = I2S_NUM_0;
    i2s_cfg.rx_handle = g_i2s_rx_handle;
    i2s_cfg.tx_handle = g_i2s_tx_handle;
    
    g_codec_data_if = audio_codec_new_i2s_data(&i2s_cfg);
    if (!g_codec_data_if) {
        ESP_LOGW(TAG, "Failed to create I2S data interface");
        return ESP_FAIL;
    }
    
    // Step 3: Create GPIO Interface (for PA control)
    if (BOARD_AUDIO_PA_PIN_GPIO >= 0) {
        g_codec_gpio_if = audio_codec_new_gpio();
    }
    
    // Step 4: Create ES8311 Codec Instance
    es8311_codec_cfg_t es8311_cfg = {};
    es8311_cfg.ctrl_if = g_codec_ctrl_if;
    es8311_cfg.gpio_if = g_codec_gpio_if;
    es8311_cfg.codec_mode = ESP_CODEC_DEV_WORK_MODE_BOTH;
    es8311_cfg.pa_pin = BOARD_AUDIO_PA_PIN_GPIO;
    es8311_cfg.pa_reverted = false;
    es8311_cfg.master_mode = false;
    es8311_cfg.use_mclk = true;
    es8311_cfg.digital_mic = false;
    es8311_cfg.invert_mclk = false;
    es8311_cfg.invert_sclk = false;
    es8311_cfg.hw_gain.pa_voltage = 5.0;
    es8311_cfg.hw_gain.codec_dac_voltage = 3.3;
    
    g_codec_if = es8311_codec_new(&es8311_cfg);
    if (!g_codec_if) {
        ESP_LOGW(TAG, "Failed to create ES8311 codec instance");
        return ESP_FAIL;
    }
    
    // Step 5: Configure Sample Rate
    esp_codec_dev_sample_info_t fs = {};
    fs.bits_per_sample = BOARD_AUDIO_BITS_PER_SAMPLE;
    fs.channel = 2; // Stereo
    fs.sample_rate = BOARD_AUDIO_SAMPLE_RATE;
    
    // Step 6: Enable I2S Channels (required before codec config)
    i2s_channel_enable(g_i2s_rx_handle);
    i2s_channel_enable(g_i2s_tx_handle);
    
    // Step 7: Configure Codec
    esp_err_t ret = g_codec_if->set_fs(g_codec_if, &fs);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "ES8311 codec not responding (I2C addr 0x%02x)", ES8311_CODEC_DEFAULT_ADDR);
        ESP_LOGW(TAG, "Codec hardware may not be present or powered");
        return ret;
    }
    
    // Step 8: Apply the saved microphone gain (or the configured default).
    ret = g_codec_if->set_mic_gain(g_codec_if, g_current_mic_gain);
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Could not set startup microphone gain: %s",
                 esp_err_to_name(ret));
    }
    
    ESP_LOGI(TAG, "ES8311 codec initialized successfully");
    return ESP_OK;
}

// ============================================================================
// Audio TX Task (Playback) - Industrial Grade Zero-Copy DMA
// ============================================================================

static void audio_tx_task(void *pvParameters) {
    while (1) {
        size_t size = 0;
        int16_t *mono = (int16_t *)xRingbufferReceiveUpTo(
            g_tx_ringbuf, &size, portMAX_DELAY, 
            4000 // Max chunk size
        );
        
        if (mono) {
            int num_samples = size / sizeof(int16_t);
            
            // Convert mono to stereo (codec expects stereo)
            int16_t* stereo_buf = (int16_t*)malloc(num_samples * 2 * sizeof(int16_t));
            if (stereo_buf) {
                for (int i = 0; i < num_samples; i++) {
                    stereo_buf[i * 2] = mono[i];
                    stereo_buf[i * 2 + 1] = mono[i];
                }
                
                size_t bytes_written = 0;
                i2s_channel_write(g_i2s_tx_handle, stereo_buf, 
                                num_samples * 2 * sizeof(int16_t), 
                                &bytes_written, portMAX_DELAY);
                
                free(stereo_buf);
            }
            
            vRingbufferReturnItem(g_tx_ringbuf, (void *)mono);
        }
    }
}

// ============================================================================
// Audio RX Task (Capture) - Continuous I2S Read with Mic Switching
// ============================================================================

static void audio_rx_task(void *pvParameters) {
    const size_t READ_CHUNK_SAMPLES = 400; // 25ms chunks @ 16kHz
    const size_t STEREO_BYTES = READ_CHUNK_SAMPLES * 2 * sizeof(int16_t);
    int16_t* stereo_buf = (int16_t*)malloc(STEREO_BYTES);
    int16_t* mono_buf = (int16_t*)malloc(READ_CHUNK_SAMPLES * sizeof(int16_t));
    
    while (1) {
        size_t bytes_read = 0;
        i2s_chan_handle_t read_handle = (g_active_mic == MIC_I2S1_INMP441) 
                                        ? g_i2s1_rx_handle 
                                        : g_i2s_rx_handle;
        
        // CRITICAL: Always read from I2S to prevent DMA buffer overflow!
        esp_err_t ret = i2s_channel_read(read_handle, stereo_buf, STEREO_BYTES, 
                                         &bytes_read, portMAX_DELAY);
        
        if (ret == ESP_OK && g_capture_running) {
            int valid_stereo_samples = bytes_read / sizeof(int16_t);
            int valid_mono_samples = valid_stereo_samples / 2;
            
            // Extract mono from stereo (left channel)
            for (int i = 0; i < valid_mono_samples; i++) {
                mono_buf[i] = stereo_buf[i * 2];
            }
            
            // Push to RX ringbuffer (non-blocking to prevent I2S stall)
            xRingbufferSend(g_rx_ringbuf, mono_buf, 
                          valid_mono_samples * sizeof(int16_t), 
                          pdMS_TO_TICKS(10));
        }
    }
}

// ============================================================================
// Public API Implementation - Industrial Grade
// ============================================================================

bool audio_manager_is_initialized(void) {
    return g_initialized;
}

esp_err_t audio_manager_init(void) {
    if (g_initialized) return ESP_OK;
    
    ESP_LOGI(TAG, "Initializing audio subsystem...");
    load_audio_settings();
    
    // Step 1: Initialize I2C Bus
    if (audio_i2c_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2C initialization failed");
        return ESP_FAIL;
    }
    
    // Step 2: Initialize I2S0 (ES8311 Codec)
    if (audio_i2s0_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2S0 initialization failed");
        return ESP_FAIL;
    }
    
    // Step 3: Initialize I2S1 (INMP441 Mic)
    if (audio_i2s1_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2S1 initialization failed");
        return ESP_FAIL;
    }
    
    // Step 4: Initialize ES8311 Codec (non-fatal for INMP441-only operation)
    if (audio_codec_init() != ESP_OK) {
        ESP_LOGW(TAG, "ES8311 codec initialization failed");
        ESP_LOGW(TAG, "Continuing with INMP441 microphone only");
        // Don't return - we can still use INMP441!
    }
    
    // Step 5: Create Ring Buffers
    g_tx_ringbuf = xRingbufferCreate(BOARD_AUDIO_TX_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    g_rx_ringbuf = xRingbufferCreate(BOARD_AUDIO_RX_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    
    if (!g_tx_ringbuf || !g_rx_ringbuf) {
        ESP_LOGE(TAG, "Failed to create ring buffers");
        return ESP_FAIL;
    }
    
    // Step 6: Create Audio Tasks (Core 1 for audio, Core 0 for VAD)
    xTaskCreatePinnedToCore(audio_tx_task, "audio_tx", 4096, NULL, 15, 
                          &g_tx_task_handle, 1);
    xTaskCreatePinnedToCore(audio_rx_task, "audio_rx", 4096, NULL, 22, 
                          &g_rx_task_handle, 1);
    
    // Step 7: Apply the saved speaker setting (or the configured default).
    if (g_codec_if) {
        esp_err_t ret = apply_speaker_volume(g_current_speaker_vol);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Could not set startup speaker volume: %s",
                     esp_err_to_name(ret));
        }
    }
    
    g_initialized = true;
    ESP_LOGI(TAG, "Audio subsystem initialized (microphone: %s, mic gain: %.0f dB, speaker volume: %u%%)",
             g_active_mic == MIC_I2S1_INMP441 ? "external INMP441" : "onboard ES8311",
             g_current_mic_gain,
             g_current_speaker_vol);
    return ESP_OK;
}

esp_err_t audio_manager_set_mic(mic_type_t mic) {
    if (mic != MIC_I2S0_ES8311 && mic != MIC_I2S1_INMP441) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!g_initialized) return ESP_ERR_INVALID_STATE;
    if (g_capture_running) return ESP_ERR_INVALID_STATE;

    g_active_mic = mic;
    const uint8_t saved_input = mic == MIC_I2S0_ES8311 ? 0 : 1;
    esp_err_t ret = save_audio_setting(AUDIO_SETTING_MIC_INPUT, saved_input);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Active mic: %s (saved)",
                 mic == MIC_I2S1_INMP441 ? "external INMP441" : "onboard ES8311");
    } else {
        ESP_LOGE(TAG, "Microphone selected but could not save it: %s",
                 esp_err_to_name(ret));
    }
    return ret;
}

mic_type_t audio_manager_get_mic(void) {
    return g_active_mic;
}

esp_err_t audio_manager_set_mic_gain(float gain_db) {
    if (!isfinite(gain_db) || gain_db < 0.0f || gain_db > 42.0f ||
        fmodf(gain_db, 6.0f) != 0.0f) {
        return ESP_ERR_INVALID_ARG;
    }
    if (!g_initialized || !g_codec_if) return ESP_ERR_INVALID_STATE;
    
    esp_err_t ret = g_codec_if->set_mic_gain(g_codec_if, gain_db);
    if (ret == ESP_OK) {
        g_current_mic_gain = gain_db;
        ret = save_audio_setting(AUDIO_SETTING_MIC_GAIN, (uint8_t)gain_db);
        if (ret == ESP_OK) {
            ESP_LOGI(TAG, "Mic gain set to %.0f dB (saved)", gain_db);
        } else {
            ESP_LOGE(TAG, "Mic gain applied but could not save it: %s",
                     esp_err_to_name(ret));
        }
    }
    return ret;
}

esp_err_t audio_manager_get_mic_gain(float *gain_db) {
    if (!gain_db) return ESP_ERR_INVALID_ARG;
    *gain_db = g_current_mic_gain;
    return ESP_OK;
}

esp_err_t audio_manager_set_speaker_vol(uint8_t vol_level) {
    if (vol_level > 100) return ESP_ERR_INVALID_ARG;
    if (!g_codec_if) return ESP_FAIL;

    esp_err_t ret = apply_speaker_volume(vol_level);
    if (ret != ESP_OK) return ret;

    g_current_speaker_vol = vol_level;
    ret = save_audio_setting(AUDIO_SETTING_SPEAKER_VOLUME, vol_level);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Speaker volume applied but could not save it: %s",
                 esp_err_to_name(ret));
    }
    return ret;
}

esp_err_t audio_manager_start_capture(void) {
    if (!g_initialized) return ESP_ERR_INVALID_STATE;
    if (g_capture_running) return ESP_OK;
    
    // Enable codec if available
    if (g_codec_if) {
        esp_err_t ret = g_codec_if->enable(g_codec_if, true);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Codec enable failed, using INMP441 only");
        }
    }
    
    g_capture_running = true;
    
    // Clear old data in RX ringbuffer
    size_t dummy;
    void* d = xRingbufferReceiveUpTo(g_rx_ringbuf, &dummy, 0, BOARD_AUDIO_RX_RINGBUF_SIZE);
    if (d) vRingbufferReturnItem(g_rx_ringbuf, d);
    
    ESP_LOGI(TAG, "Audio capture started");
    return ESP_OK;
}

esp_err_t audio_manager_stop_capture(void) {
    if (!g_capture_running) return ESP_OK;
    
    // Disable codec only if playback is also stopped
    if (g_codec_if && !g_playback_running) {
        g_codec_if->enable(g_codec_if, false);
    }
    
    g_capture_running = false;
    ESP_LOGI(TAG, "Audio capture stopped");
    return ESP_OK;
}

bool audio_manager_is_capturing(void) {
    return g_capture_running;
}

int audio_manager_read(int16_t *buffer, int num_samples, int timeout_ms) {
    if (!g_capture_running || !buffer || num_samples <= 0) return 0;
    
    size_t requested_bytes = num_samples * sizeof(int16_t);
    size_t total_received_bytes = 0;
    uint32_t start_time = xTaskGetTickCount();
    
    while (total_received_bytes < requested_bytes) {
        if (!g_capture_running) break;
        
        uint32_t elapsed = (xTaskGetTickCount() - start_time) * portTICK_PERIOD_MS;
        if (elapsed >= timeout_ms && timeout_ms != portMAX_DELAY) {
            break; // Timeout
        }
        
        uint32_t wait_ticks = (timeout_ms == portMAX_DELAY) 
                             ? portMAX_DELAY 
                             : pdMS_TO_TICKS(timeout_ms - elapsed);
        if (wait_ticks == 0 && timeout_ms != 0) wait_ticks = 1;
        
        size_t received_bytes = 0;
        void *data = xRingbufferReceiveUpTo(g_rx_ringbuf, &received_bytes, 
                                           wait_ticks, 
                                           requested_bytes - total_received_bytes);
        
        if (data) {
            memcpy((uint8_t*)buffer + total_received_bytes, data, received_bytes);
            vRingbufferReturnItem(g_rx_ringbuf, data);
            total_received_bytes += received_bytes;
        }
    }
    
    return total_received_bytes / sizeof(int16_t);
}

esp_err_t audio_manager_start_playback(void) {
    if (!g_initialized) return ESP_ERR_INVALID_STATE;
    if (g_playback_running) return ESP_OK;
    
    // Enable codec if available
    if (g_codec_if) {
        esp_err_t ret = g_codec_if->enable(g_codec_if, true);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Codec enable for playback failed");
            return ret;
        }
        ret = apply_speaker_volume(g_current_speaker_vol);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Could not restore speaker volume %u%% for playback: %s",
                     g_current_speaker_vol, esp_err_to_name(ret));
            return ret;
        }
    }
    
    g_playback_running = true;
    ESP_LOGI(TAG, "Audio playback started");
    return ESP_OK;
}

esp_err_t audio_manager_stop_playback(void) {
    if (!g_playback_running) return ESP_OK;
    
    // Wait for TX ringbuffer to drain (max 3 seconds)
    uint32_t wait_ticks = 300; // 3000ms / 10ms
    while (xRingbufferGetCurFreeSize(g_tx_ringbuf) < BOARD_AUDIO_TX_RINGBUF_SIZE && wait_ticks > 0) {
        vTaskDelay(pdMS_TO_TICKS(10));
        wait_ticks--;
    }
    
    g_playback_running = false;
    
    // Disable codec only if capture is also stopped
    if (g_codec_if && !g_capture_running) {
        g_codec_if->enable(g_codec_if, false);
    }
    
    ESP_LOGI(TAG, "Audio playback stopped");
    return ESP_OK;
}

bool audio_manager_is_playing(void) {
    return g_playback_running;
}

int audio_manager_write(const int16_t *buffer, int num_samples, int timeout_ms) {
    if (!g_playback_running || !buffer || num_samples <= 0) return 0;
    
    size_t size = num_samples * sizeof(int16_t);
    BaseType_t res = xRingbufferSend(g_tx_ringbuf, buffer, size, 
                                     pdMS_TO_TICKS(timeout_ms));
    
    if (res != pdTRUE) return -1;
    return num_samples;
}

void audio_manager_play_boot_sequence(void) {
    if (!g_initialized) return;
    
    const float freqs[] = {261.63f, 329.63f, 392.00f, 523.25f}; // C-E-G-C
    const int duration_ms = 150;
    const int sample_rate = BOARD_AUDIO_SAMPLE_RATE;
    const float amplitude = 8000.0f;
    
    audio_manager_start_playback();
    
    for (int note = 0; note < 4; note++) {
        int total_samples = (sample_rate * duration_ms) / 1000;
        int16_t *buffer = (int16_t *)malloc(total_samples * sizeof(int16_t));
        if (!buffer) break;
        
        for (int i = 0; i < total_samples; i++) {
            float t = (float)i / sample_rate;
            buffer[i] = (int16_t)(amplitude * sin(2.0f * M_PI * freqs[note] * t));
        }
        
        audio_manager_write(buffer, total_samples, BOARD_AUDIO_I2S_READ_TIMEOUT_MS);
        free(buffer);
    }
    
    audio_manager_stop_playback();
}
