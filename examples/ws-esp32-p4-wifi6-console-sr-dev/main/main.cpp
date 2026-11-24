/**
 * @file main.cpp
 * @brief Waveshare ESP32-P4-WIFI6 Native Speaker Recognition Console
 */

#include <cstdio>
#include <cstring>
#include <errno.h>
#include <sys/stat.h>
#include "esp_log.h"
#include "esp_console.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "nvs_flash.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "sdmmc_cmd.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "soc/soc_caps.h"
#include "esp_private/esp_clk.h"

#include "app_config.h"
#include "audio_manager.h"
#include "cmd_sr_cli.h"
#include "wifi_transfer.h"
#include "esp_speaker_recognition.h"

static const char* TAG = APP_TAG;

// ============================================================================
// SD Card Mounting (Waveshare ESP32-P4-WIFI6 MicroSD via SDMMC Slot 0)
// ============================================================================
#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
static esp_err_t shared_sdmmc_host_init(void) { return ESP_OK; }
static esp_err_t shared_sdmmc_host_deinit(void) { return ESP_OK; }
#endif

static esp_err_t mount_sd_card(void) {
#if SOC_SDMMC_HOST_SUPPORTED
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    host.slot = SDMMC_HOST_SLOT_0;
#endif
#if defined(CONFIG_IDF_TARGET_ESP32P4) && \
    defined(CONFIG_ESP_HOSTED_HOST_TRANSPORT_BUS_SDIO) && \
    ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    host.init = shared_sdmmc_host_init;
    host.deinit = shared_sdmmc_host_deinit;
#endif

#if defined(CONFIG_IDF_TARGET_ESP32P4)
    sd_pwr_ctrl_ldo_config_t ldo_config = {.ldo_chan_id = 4};
    sd_pwr_ctrl_handle_t power_control = nullptr;
    const esp_err_t power_error = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &power_control);
    if (power_error != ESP_OK) {
        return power_error;
    }
    host.pwr_ctrl_handle = power_control;
#endif

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    slot.clk = static_cast<gpio_num_t>(43);
    slot.cmd = static_cast<gpio_num_t>(44);
    slot.d0 = static_cast<gpio_num_t>(39);
    slot.d1 = static_cast<gpio_num_t>(40);
    slot.d2 = static_cast<gpio_num_t>(41);
    slot.d3 = static_cast<gpio_num_t>(42);
    slot.width = 4;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;
#endif

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 10;
    mount_config.allocation_unit_size = 16 * 1024;
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    mount_config.disk_status_check_enable = true;
    mount_config.use_one_fat = false;
#endif

    static sdmmc_card_t *card = nullptr;
    const esp_err_t error = esp_vfs_fat_sdmmc_mount(
        FS_MOUNT_POINT_SD, &host, &slot, &mount_config, &card);
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "SD-card mount failed: %s", esp_err_to_name(error));
#if defined(CONFIG_IDF_TARGET_ESP32P4)
        sd_pwr_ctrl_del_on_chip_ldo(power_control);
#endif
        return error;
    }

    sdmmc_card_print_info(stdout, card);
    const char *directories[] = {"/sd/models", "/sd/audio", "/sd/results"};
    for (const char *directory : directories) {
        if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
            ESP_LOGE(TAG, "Cannot create %s: errno=%d", directory, errno);
            return ESP_FAIL;
        }
    }
    ESP_LOGI(TAG, "SD card mounted at %s", FS_MOUNT_POINT_SD);
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

// ============================================================================
// Banner & System Info
// ============================================================================
static void print_banner(esp_err_t sd_status) {
    size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t total_sram = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    uint32_t cpu_mhz = esp_clk_cpu_freq() / 1000000;

    printf("\n");
    printf("===================================================================\n");
    printf("  ESP32-P4 NATIVE SPEAKER RECOGNITION CONSOLE\n");
    printf("  WeSpeaker ECAPA-TDNN512 SOTA | 100%% RV32 PIE SIMD | Zero esp-dl\n");
    printf("===================================================================\n");
    printf("  Hardware : Waveshare ESP32-P4-WIFI6 (Dual-Core @ %u MHz)\n", (unsigned)cpu_mhz);
    printf("  PSRAM    : %u / %u MB free\n", (unsigned)(free_psram / (1024*1024)), (unsigned)(total_psram / (1024*1024)));
    printf("  SRAM     : %u / %u KB free\n", (unsigned)(free_sram / 1024), (unsigned)(total_sram / 1024));
    if (sd_status == ESP_OK) {
        printf("  SD Card  : %s (mounted)\n", FS_MOUNT_POINT_SD);
        printf("  Audio    : %s/audio/*.wav\n", FS_MOUNT_POINT_SD);
    } else {
        printf("  SD Card  : unavailable (%s)\n", esp_err_to_name(sd_status));
    }
    printf("  Commands : info, sr_profile, sr_enroll, sr_verify, sr_identify,\n");
    printf("             sr_list, sr_delete, sr_clear, sr_bench,\n");
    printf("             mic, speaker_vol, play_wav, record_mic, wifi\n");
    printf("===================================================================\n\n");
}

extern "C" void app_main(void) {
    // 1. Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 2. Mount MicroSD Card
    esp_err_t sd_status = mount_sd_card();

    // 3. Print Banner
    print_banner(sd_status);

    // 4. Self-test PIE INT8 SIMD Kernel with aligned buffers
    int8_t test_a[1536] __attribute__((aligned(16)));
    int8_t test_b[1536] __attribute__((aligned(16)));
    for (int i = 0; i < 1536; i++) {
        test_a[i] = (int8_t)((i % 25) - 12);
        test_b[i] = (int8_t)(((i * 3) % 25) - 12);
    }
    int32_t hw_sum = 0, ref_sum = 0;
    uint32_t c0 = esp_cpu_get_cycle_count();
    sr_dotprod_s8_pie(test_a, test_b, &hw_sum, 1536);
    uint32_t c1 = esp_cpu_get_cycle_count();
    for (int i = 0; i < 1536; i++) ref_sum += (int32_t)test_a[i] * (int32_t)test_b[i];

    // Test with PSRAM buffer for test_a
    void *psram_raw = heap_caps_malloc(1536 + 32, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *psram_a = (int8_t *)(((uintptr_t)psram_raw + 15) & ~15);
    memcpy(psram_a, test_a, 1536);
    int32_t psram_sum = 0;
    uint32_t c2 = esp_cpu_get_cycle_count();
    sr_dotprod_s8_pie(psram_a, test_b, &psram_sum, 1536);
    uint32_t c3 = esp_cpu_get_cycle_count();
    free(psram_raw);

    ESP_LOGI(TAG, "PIE INT8 SIMD (1536-elem): HW=%ld, Ref=%ld [%s], SRAM cycles=%lu, PSRAM cycles=%lu",
             (long)hw_sum, (long)ref_sum, (hw_sum == ref_sum) ? "PASSED" : "FAILED",
             (unsigned long)(c1 - c0), (unsigned long)(c3 - c2));

    // 5. Initialize Native Speaker Recognition from Flash Partition
    ret = esp_speaker_recognition_init_partition(SR_PARTITION_LABEL, SR_DEFAULT_THRESHOLD);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize speaker recognition: %s", esp_err_to_name(ret));
    } else {
        ESP_LOGI(TAG, "Speaker Recognition online: %u-D ECAPA-TDNN512, Threshold: %.2f",
                 esp_speaker_recognition_get_embedding_dim(), SR_DEFAULT_THRESHOLD);
    }

    // 6. Initialize Board Audio (ES8311 + INMP441)
    ESP_LOGI(TAG, "Initializing Audio Hardware...");
    ret = audio_manager_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Audio hardware init returned: %s", esp_err_to_name(ret));
    }

    // 7. Register Console Commands
    esp_console_repl_t *repl = nullptr;
    esp_console_repl_config_t repl_config = ESP_CONSOLE_REPL_CONFIG_DEFAULT();
    repl_config.prompt = "esp32p4-sr> ";
    repl_config.max_cmdline_length = 256;
    repl_config.task_stack_size = 16384;

    esp_console_dev_uart_config_t uart_config = ESP_CONSOLE_DEV_UART_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_console_new_repl_uart(&uart_config, &repl_config, &repl));

    // Register CLI modules
    cmd_sr_cli_register();
    wifi_transfer_register_commands();

    ESP_LOGI(TAG, "Console ready. Type 'help' for available commands.");
    ESP_ERROR_CHECK(esp_console_start_repl(repl));
}
