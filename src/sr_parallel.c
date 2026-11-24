/**
 * @file sr_parallel.c
 * @brief Dual-core worker implementation using FreeRTOS task notifications for ESP32-P4
 */

#include "sr_parallel.h"
#include "esp_log.h"
#include "sdkconfig.h"

#if defined(CONFIG_FREERTOS_NUMBER_OF_CORES) && (CONFIG_FREERTOS_NUMBER_OF_CORES > 1)

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

static const char *TAG = "SR_PAR";

static TaskHandle_t s_worker = NULL;
static SemaphoreHandle_t s_done = NULL;
static volatile sr_par_fn s_job_fn = NULL;
static void *volatile s_job_ctx = NULL;
static volatile bool s_running = false;

static void worker_task(void *arg) {
    (void)arg;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (!s_running) {
            break;
        }
        if (s_job_fn) {
            s_job_fn(s_job_ctx, 1, SR_PARALLEL_PARTS);
        }
        xSemaphoreGive(s_done);
    }
    vTaskDelete(NULL);
}

void sr_parallel_init(void) {
    if (!s_worker) {
        s_done = xSemaphoreCreateBinary();
        if (!s_done) {
            ESP_LOGE(TAG, "Failed to create completion semaphore");
            return;
        }
        s_running = true;
        BaseType_t ret = xTaskCreatePinnedToCore(
            worker_task,
            "sr_worker_c1",
            16384,
            NULL,
            configMAX_PRIORITIES - 2,
            &s_worker,
            1
        );
        if (ret == pdPASS) {
            ESP_LOGI(TAG, "Dual-core worker online (pinned to Core 1)");
        } else {
            ESP_LOGE(TAG, "Failed to create worker task on Core 1");
            vSemaphoreDelete(s_done);
            s_done = NULL;
            s_worker = NULL;
            s_running = false;
        }
    }
}

void sr_parallel_deinit(void) {
    if (s_worker) {
        s_running = false;
        xTaskNotifyGive(s_worker);
        vTaskDelay(pdMS_TO_TICKS(10));
        s_worker = NULL;
        if (s_done) {
            vSemaphoreDelete(s_done);
            s_done = NULL;
        }
        ESP_LOGI(TAG, "Dual-core worker deinitialized");
    }
}

bool sr_parallel_is_active(void) {
    return (s_worker != NULL && s_running);
}

void sr_parallel_run(sr_par_fn fn, void *ctx) {
    if (!s_worker || !s_running) {
        fn(ctx, 0, SR_PARALLEL_PARTS);
        fn(ctx, 1, SR_PARALLEL_PARTS);
        return;
    }
    s_job_fn = fn;
    s_job_ctx = ctx;
    xTaskNotifyGive(s_worker);
    fn(ctx, 0, SR_PARALLEL_PARTS);
    xSemaphoreTake(s_done, portMAX_DELAY);
}

#else

void sr_parallel_init(void) {}
void sr_parallel_deinit(void) {}
bool sr_parallel_is_active(void) { return false; }
void sr_parallel_run(sr_par_fn fn, void *ctx) {
    fn(ctx, 0, SR_PARALLEL_PARTS);
    fn(ctx, 1, SR_PARALLEL_PARTS);
}

#endif
