#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define APP_TAG "SR_CONSOLE"

// Audio Configuration
#define SR_SAMPLE_RATE          16000
#define SR_CHANNELS             1
#define SR_BITS_PER_SAMPLE      16

// Model Partition and Verification Threshold
#define SR_PARTITION_LABEL      "model"
#define SR_DEFAULT_THRESHOLD    0.30f

// Storage Configuration
#define FS_MOUNT_POINT_SD       "/sd"
#define STORAGE_MOUNT_POINT     "/sd"
#define STORAGE_PARTITION_LABEL "storage"
#define MAX_AUDIO_DURATION_SEC  300

#ifdef __cplusplus
}
#endif
