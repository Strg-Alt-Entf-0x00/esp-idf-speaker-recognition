#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t wifi_transfer_init(void);
void wifi_transfer_register_commands(void);

#ifdef __cplusplus
}
#endif
