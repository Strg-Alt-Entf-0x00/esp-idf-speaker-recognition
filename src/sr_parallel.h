#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SR_PARALLEL_PARTS 2

typedef void (*sr_par_fn)(void *ctx, int part_idx, int total_parts);

/**
 * @brief Initialize the dual-core worker task pinned to Core 1.
 */
void sr_parallel_init(void);

/**
 * @brief Deinitialize the dual-core worker.
 */
void sr_parallel_deinit(void);

/**
 * @brief Check if parallel worker is active.
 */
bool sr_parallel_is_active(void);

/**
 * @brief Run a partitioned job across Core 0 and Core 1 in parallel.
 *
 * Core 0 executes part 0, Core 1 executes part 1.
 * Blocks until both cores have completed.
 */
void sr_parallel_run(sr_par_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif
