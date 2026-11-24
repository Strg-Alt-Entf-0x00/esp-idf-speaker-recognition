#include "wifi_transfer.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>

#include "driver/uart.h"
#include "esp_console.h"
#include "esp_event.h"
#include "esp_hosted.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_random.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

namespace {

constexpr char TAG[] = "wifi_transfer";
constexpr char NVS_NAMESPACE[] = "wifi_cfg";
constexpr char NVS_SSID_KEY[] = "ssid";
constexpr char NVS_PASSWORD_KEY[] = "password";
constexpr char TOKEN_PREFIX[] = "Bearer ";
constexpr size_t TOKEN_BYTES = 16;
constexpr size_t TOKEN_HEX_LENGTH = TOKEN_BYTES * 2;
constexpr size_t MAX_FILENAME_LENGTH = 80;
constexpr size_t MAX_UPLOAD_BYTES = 64U * 1024U * 1024U;
constexpr int WIFI_RETRY_LIMIT = 5;

std::atomic<bool> s_connected{false};
std::atomic<bool> s_disconnect_requested{false};
std::atomic<int> s_retry_count{0};
bool s_netif_ready = false;
bool s_wifi_ready = false;
bool s_wifi_started = false;
bool s_hosted_info_logged = false;
bool s_sd_available = false;
SemaphoreHandle_t s_server_mutex = nullptr;
esp_netif_t *s_sta_netif = nullptr;
esp_event_handler_instance_t s_wifi_handler = nullptr;
esp_event_handler_instance_t s_ip_handler = nullptr;
bool s_handlers_registered = false;
httpd_handle_t s_http_server = nullptr;
char s_transfer_token[TOKEN_HEX_LENGTH + 1] = {};

struct credentials_t {
    char ssid[33];
    char password[65];
};

static esp_err_t send_http_error(httpd_req_t *request, const char *status, const char *message)
{
    httpd_resp_set_status(request, status);
    return httpd_resp_send(request, message, HTTPD_RESP_USE_STRLEN);
}

static bool has_valid_token(httpd_req_t *request)
{
    char authorization[sizeof(TOKEN_PREFIX) + TOKEN_HEX_LENGTH] = {};
    const size_t header_length = httpd_req_get_hdr_value_len(request, "Authorization");
    if (header_length == 0 || header_length >= sizeof(authorization)) {
        return false;
    }

    if (httpd_req_get_hdr_value_str(
            request, "Authorization", authorization, sizeof(authorization)) != ESP_OK) {
        return false;
    }

    char expected[sizeof(TOKEN_PREFIX) + TOKEN_HEX_LENGTH] = {};
    const int expected_length = snprintf(
        expected, sizeof(expected), "%s%s", TOKEN_PREFIX, s_transfer_token);
    return expected_length > 0 &&
           static_cast<size_t>(expected_length) < sizeof(expected) &&
           strcmp(authorization, expected) == 0;
}

static bool valid_filename(const char *filename)
{
    if (!filename || filename[0] == '\0' || filename[0] == '.' ||
        strlen(filename) > MAX_FILENAME_LENGTH) {
        return false;
    }

    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(filename); *p; ++p) {
        if (!(isalnum(*p) || *p == '.' || *p == '_' || *p == '-')) {
            return false;
        }
    }
    return true;
}

static bool allowed_extension(const char *category, const char *filename)
{
    const char *extension = strrchr(filename, '.');
    if (!extension) {
        return false;
    }

    if (strcmp(category, "models") == 0) {
        return strcmp(extension, ".espm") == 0;
    }
    if (strcmp(category, "audio") == 0) {
        return strcmp(extension, ".wav") == 0;
    }
    if (strcmp(category, "results") == 0) {
        return strcmp(extension, ".csv") == 0 ||
               strcmp(extension, ".json") == 0 ||
               strcmp(extension, ".bin") == 0;
    }
    return false;
}

static bool resolve_file_uri(const char *uri, char *category, size_t category_capacity,
                             char *filename, size_t filename_capacity, char *path,
                             size_t path_capacity)
{
    constexpr char prefix[] = "/api/";
    if (!uri || strncmp(uri, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }

    const char *category_start = uri + sizeof(prefix) - 1;
    const char *separator = strchr(category_start, '/');
    if (!separator || separator == category_start || strchr(separator + 1, '/')) {
        return false;
    }

    const size_t category_length = static_cast<size_t>(separator - category_start);
    const char *name = separator + 1;
    if (category_length >= category_capacity ||
        strlen(name) >= filename_capacity ||
        !valid_filename(name)) {
        return false;
    }

    memcpy(category, category_start, category_length);
    category[category_length] = '\0';
    memcpy(filename, name, strlen(name) + 1);
    if (!allowed_extension(category, filename)) {
        return false;
    }

    const char *directory = nullptr;
    if (strcmp(category, "models") == 0) {
        directory = "/sd/models";
    } else if (strcmp(category, "audio") == 0) {
        directory = "/sd/audio";
    } else if (strcmp(category, "results") == 0) {
        directory = "/sd/results";
    } else {
        return false;
    }

    const int written = snprintf(path, path_capacity, "%s/%s", directory, filename);
    return written > 0 && static_cast<size_t>(written) < path_capacity;
}

static esp_err_t file_get_handler(httpd_req_t *request)
{
    if (!has_valid_token(request)) {
        return send_http_error(request, "401 Unauthorized", "Transfer token required");
    }
    if (!s_sd_available) {
        return send_http_error(request, "503 Service Unavailable", "SD card is not mounted");
    }

    char category[16] = {};
    char filename[MAX_FILENAME_LENGTH + 1] = {};
    char path[160] = {};
    if (!resolve_file_uri(request->uri, category, sizeof(category),
                          filename, sizeof(filename), path, sizeof(path))) {
        return send_http_error(request, "400 Bad Request", "Invalid file path");
    }

    FILE *file = fopen(path, "rb");
    if (!file) {
        return send_http_error(request, "404 Not Found", "File not found");
    }

    const char *content_type = "application/octet-stream";
    if (strcmp(category, "audio") == 0) {
        content_type = "audio/wav";
    } else if (strcmp(category, "results") == 0 &&
               strcmp(strrchr(filename, '.'), ".csv") == 0) {
        content_type = "text/csv";
    } else if (strcmp(strrchr(filename, '.'), ".json") == 0) {
        content_type = "application/json";
    }

    char disposition[128];
    esp_err_t error = httpd_resp_set_type(request, content_type);
    if (error == ESP_OK) {
        const int length = snprintf(
            disposition, sizeof(disposition), "attachment; filename=\"%s\"", filename);
        if (length <= 0 || static_cast<size_t>(length) >= sizeof(disposition)) {
            fclose(file);
            return send_http_error(request, "500 Internal Server Error", "Response setup failed");
        }
        error = httpd_resp_set_hdr(request, "Content-Disposition", disposition);
    }

    char buffer[4096];
    while (error == ESP_OK) {
        const size_t count = fread(buffer, 1, sizeof(buffer), file);
        if (count > 0) {
            error = httpd_resp_send_chunk(request, buffer, count);
        }
        if (count < sizeof(buffer)) {
            if (ferror(file)) {
                error = ESP_FAIL;
            }
            break;
        }
    }
    fclose(file);

    if (error == ESP_OK) {
        error = httpd_resp_send_chunk(request, nullptr, 0);
    }
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "File download failed: %s", esp_err_to_name(error));
    }
    return error;
}

static esp_err_t file_put_handler(httpd_req_t *request)
{
    if (!has_valid_token(request)) {
        return send_http_error(request, "401 Unauthorized", "Transfer token required");
    }
    if (!s_sd_available) {
        return send_http_error(request, "503 Service Unavailable", "SD card is not mounted");
    }
    if (request->content_len == 0 || request->content_len > MAX_UPLOAD_BYTES) {
        return send_http_error(
            request,
            request->content_len == 0 ? "400 Bad Request" : "413 Payload Too Large",
            "Invalid upload size");
    }

    char category[16] = {};
    char filename[MAX_FILENAME_LENGTH + 1] = {};
    char path[160] = {};
    if (!resolve_file_uri(request->uri, category, sizeof(category),
                          filename, sizeof(filename), path, sizeof(path))) {
        return send_http_error(request, "400 Bad Request", "Invalid file path");
    }

    struct stat status = {};
    if (stat(path, &status) == 0) {
        return send_http_error(request, "409 Conflict", "File already exists");
    }
    if (errno != ENOENT) {
        return send_http_error(request, "500 Internal Server Error", "Cannot inspect destination");
    }

    char directory[32] = {};
    const int directory_length = snprintf(directory, sizeof(directory), "%.*s",
        static_cast<int>(strrchr(path, '/') - path), path);
    if (directory_length <= 0 || static_cast<size_t>(directory_length) >= sizeof(directory)) {
        return send_http_error(request, "500 Internal Server Error", "Invalid destination");
    }
    if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
        ESP_LOGE(TAG, "Cannot create destination directory: errno=%d", errno);
        return send_http_error(request, "500 Internal Server Error", "Cannot create destination");
    }

    char temporary_path[176] = {};
    const int temporary_length = snprintf(
        temporary_path, sizeof(temporary_path), "%s.part", path);
    if (temporary_length <= 0 ||
        static_cast<size_t>(temporary_length) >= sizeof(temporary_path)) {
        return send_http_error(request, "500 Internal Server Error", "Invalid temporary path");
    }

    const int fd = open(temporary_path, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        return send_http_error(request, errno == EEXIST ? "409 Conflict" : "500 Internal Server Error",
                               errno == EEXIST ? "An upload with this name is in progress" :
                                                 "Cannot create upload file");
    }

    char buffer[4096];
    size_t remaining = request->content_len;
    esp_err_t error = ESP_OK;
    while (remaining > 0) {
        const size_t wanted = remaining < sizeof(buffer) ? remaining : sizeof(buffer);
        const int received = httpd_req_recv(request, buffer, wanted);
        if (received <= 0) {
            error = received == HTTPD_SOCK_ERR_TIMEOUT ? ESP_ERR_TIMEOUT : ESP_FAIL;
            break;
        }

        size_t written = 0;
        while (written < static_cast<size_t>(received)) {
            const ssize_t count = write(fd, buffer + written,
                                        static_cast<size_t>(received) - written);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                error = ESP_FAIL;
                break;
            }
            written += static_cast<size_t>(count);
        }
        if (error != ESP_OK) {
            break;
        }
        remaining -= static_cast<size_t>(received);
    }

    if (error == ESP_OK && fsync(fd) != 0) {
        error = ESP_FAIL;
    }
    if (close(fd) != 0 && error == ESP_OK) {
        error = ESP_FAIL;
    }
    if (error == ESP_OK && rename(temporary_path, path) != 0) {
        error = ESP_FAIL;
    }
    if (error != ESP_OK) {
        unlink(temporary_path);
        ESP_LOGE(TAG, "File upload failed: %s", esp_err_to_name(error));
        return send_http_error(request, "500 Internal Server Error", "Upload failed");
    }

    ESP_LOGI(TAG, "Received %s file (%u bytes)", category,
             static_cast<unsigned>(request->content_len));
    httpd_resp_set_status(request, "201 Created");
    return httpd_resp_send(request, "Upload complete", HTTPD_RESP_USE_STRLEN);
}

static void stop_file_server_locked(void)
{
    if (s_http_server) {
        const esp_err_t error = httpd_stop(s_http_server);
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "Cannot stop file server: %s", esp_err_to_name(error));
        } else {
            s_http_server = nullptr;
        }
    }
}

static void stop_file_server(void)
{
    if (!s_server_mutex) {
        return;
    }
    xSemaphoreTake(s_server_mutex, portMAX_DELAY);
    stop_file_server_locked();
    xSemaphoreGive(s_server_mutex);
}

static esp_err_t start_file_server(void)
{
    if (!s_server_mutex) {
        return ESP_ERR_INVALID_STATE;
    }
    xSemaphoreTake(s_server_mutex, portMAX_DELAY);
    if (s_http_server) {
        xSemaphoreGive(s_server_mutex);
        return ESP_OK;
    }
    if (!s_sd_available) {
        ESP_LOGW(TAG, "File transfer is unavailable because the SD card is not mounted");
        xSemaphoreGive(s_server_mutex);
        return ESP_ERR_INVALID_STATE;
    }

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 8080;
    config.stack_size = 8192;
    config.max_uri_handlers = 4;
    config.recv_wait_timeout = 30;
    config.send_wait_timeout = 30;
    config.uri_match_fn = httpd_uri_match_wildcard;

    esp_err_t error = httpd_start(&s_http_server, &config);
    if (error != ESP_OK) {
        s_http_server = nullptr;
        ESP_LOGE(TAG, "Cannot start file server: %s", esp_err_to_name(error));
        xSemaphoreGive(s_server_mutex);
        return error;
    }

    httpd_uri_t get_uri = {};
    get_uri.uri = "/api/*";
    get_uri.method = HTTP_GET;
    get_uri.handler = file_get_handler;
    error = httpd_register_uri_handler(s_http_server, &get_uri);
    if (error == ESP_OK) {
        httpd_uri_t put_uri = {};
        put_uri.uri = "/api/*";
        put_uri.method = HTTP_PUT;
        put_uri.handler = file_put_handler;
        error = httpd_register_uri_handler(s_http_server, &put_uri);
    }
    if (error != ESP_OK) {
        stop_file_server_locked();
        ESP_LOGE(TAG, "Cannot register file transfer routes: %s", esp_err_to_name(error));
        xSemaphoreGive(s_server_mutex);
        return error;
    }

    ESP_LOGI(TAG, "Authenticated SD-card transfer service started on port 8080");
    xSemaphoreGive(s_server_mutex);
    return ESP_OK;
}

static void wifi_event_handler(void *, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base != WIFI_EVENT) {
        return;
    }

    if (event_id == WIFI_EVENT_STA_START) {
        const esp_err_t error = esp_wifi_connect();
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "Wi-Fi connection request failed: %s", esp_err_to_name(error));
        }
    } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected.store(false);
        const auto *event = static_cast<const wifi_event_sta_disconnected_t *>(event_data);
        const int reason = event ? event->reason : 0;
        const int retries = s_retry_count.load();
        if (!s_disconnect_requested.load() && retries < WIFI_RETRY_LIMIT) {
            const int retry = s_retry_count.fetch_add(1) + 1;
            const esp_err_t error = esp_wifi_connect();
            if (error != ESP_OK) {
                ESP_LOGE(TAG, "Wi-Fi reconnect request failed: %s", esp_err_to_name(error));
            } else {
                ESP_LOGW(TAG, "Wi-Fi disconnected (reason %d); retry %d/%d",
                         reason, retry, WIFI_RETRY_LIMIT);
            }
        } else if (!s_disconnect_requested.load()) {
            ESP_LOGE(TAG, "Wi-Fi reconnect limit reached (reason %d); use 'wifi connect' to retry",
                     reason);
        }
    }
}

static void ip_event_handler(void *, esp_event_base_t event_base,
                             int32_t event_id, void *event_data)
{
    if (event_base != IP_EVENT || event_id != IP_EVENT_STA_GOT_IP) {
        return;
    }

    const auto *event = static_cast<const ip_event_got_ip_t *>(event_data);
    s_connected.store(true);
    s_retry_count.store(0);
    ESP_LOGI(TAG, "Wi-Fi connected; run 'wifi status' to view the transfer address");
    if (!event) {
        ESP_LOGE(TAG, "Wi-Fi IP event did not include address data");
        return;
    }

    const esp_err_t error = start_file_server();
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "Wi-Fi is connected but file transfer could not start: %s",
                 esp_err_to_name(error));
    }
}

static esp_err_t ensure_wifi_stack(void)
{
    if (!s_hosted_info_logged) {
        esp_hosted_app_desc_t coprocessor = {};
        const esp_err_t hosted_error = esp_hosted_get_coprocessor_app_desc(&coprocessor);
        if (hosted_error == ESP_OK) {
            ESP_LOGI(TAG, "C6 firmware: project=%s version=%s IDF=%s",
                     coprocessor.project_name, coprocessor.version, coprocessor.idf_ver);
            s_hosted_info_logged = true;
        } else {
            ESP_LOGW(TAG, "C6 firmware information unavailable: %s",
                     esp_err_to_name(hosted_error));
        }
    }

    if (!s_netif_ready) {
        esp_err_t error = esp_netif_init();
        if (error != ESP_OK && error != ESP_ERR_INVALID_STATE) {
            return error;
        }

        error = esp_event_loop_create_default();
        if (error != ESP_OK && error != ESP_ERR_INVALID_STATE) {
            return error;
        }

        s_sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
        if (!s_sta_netif) {
            s_sta_netif = esp_netif_create_default_wifi_sta();
        }
        if (!s_sta_netif) {
            return ESP_ERR_NO_MEM;
        }
        s_netif_ready = true;
    }

    if (!s_wifi_ready) {
        wifi_init_config_t config = WIFI_INIT_CONFIG_DEFAULT();
        esp_err_t error = esp_wifi_init(&config);
        if (error != ESP_OK) {
            return error;
        }
        s_wifi_ready = true;
    }

    if (!s_handlers_registered) {
        esp_err_t error = esp_event_handler_instance_register(
            WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, nullptr, &s_wifi_handler);
        if (error != ESP_OK) {
            return error;
        }
        error = esp_event_handler_instance_register(
            IP_EVENT, IP_EVENT_STA_GOT_IP, ip_event_handler, nullptr, &s_ip_handler);
        if (error != ESP_OK) {
            const esp_err_t unregister_error = esp_event_handler_instance_unregister(
                WIFI_EVENT, ESP_EVENT_ANY_ID, s_wifi_handler);
            if (unregister_error != ESP_OK) {
                ESP_LOGE(TAG, "Cannot roll back Wi-Fi event handler: %s",
                         esp_err_to_name(unregister_error));
            }
            s_wifi_handler = nullptr;
            return error;
        }
        s_handlers_registered = true;
    }
    return ESP_OK;
}

static esp_err_t start_station(const credentials_t &credentials)
{
    esp_err_t error = ensure_wifi_stack();
    if (error != ESP_OK) {
        return error;
    }

    s_disconnect_requested.store(true);
    if (s_wifi_started) {
        s_connected.store(false);
        error = esp_wifi_disconnect();
        if (error != ESP_OK && error != ESP_ERR_WIFI_NOT_CONNECT) {
            ESP_LOGW(TAG, "Wi-Fi disconnect before reconfiguration returned: %s",
                     esp_err_to_name(error));
        }
        stop_file_server();
        error = esp_wifi_stop();
        if (error != ESP_OK) {
            return error;
        }
        s_wifi_started = false;
    }

    wifi_config_t config = {};
    memcpy(config.sta.ssid, credentials.ssid, strlen(credentials.ssid));
    memcpy(config.sta.password, credentials.password, strlen(credentials.password));
    config.sta.threshold.authmode = credentials.password[0] == '\0' ?
        WIFI_AUTH_OPEN : WIFI_AUTH_WPA2_PSK;

    error = esp_wifi_set_mode(WIFI_MODE_STA);
    if (error == ESP_OK) {
        error = esp_wifi_set_config(WIFI_IF_STA, &config);
    }
    if (error != ESP_OK) {
        return error;
    }

    s_retry_count.store(0);
    s_connected.store(false);
    s_disconnect_requested.store(false);
    error = esp_wifi_start();
    if (error == ESP_OK) {
        s_wifi_started = true;
    }
    return error;
}

static esp_err_t stop_station(void)
{
    s_disconnect_requested.store(true);
    s_connected.store(false);
    stop_file_server();
    if (!s_wifi_started) {
        return ESP_OK;
    }

    esp_err_t error = esp_wifi_disconnect();
    if (error != ESP_OK && error != ESP_ERR_WIFI_NOT_CONNECT) {
        ESP_LOGW(TAG, "Wi-Fi disconnect returned: %s", esp_err_to_name(error));
    }
    error = esp_wifi_stop();
    if (error == ESP_OK) {
        s_wifi_started = false;
    }
    return error;
}

static esp_err_t save_credentials(const credentials_t &credentials)
{
    nvs_handle_t handle;
    esp_err_t error = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (error != ESP_OK) {
        return error;
    }

    error = nvs_set_str(handle, NVS_SSID_KEY, credentials.ssid);
    if (error == ESP_OK) {
        error = nvs_set_str(handle, NVS_PASSWORD_KEY, credentials.password);
    }
    if (error == ESP_OK) {
        error = nvs_commit(handle);
    }
    nvs_close(handle);
    return error;
}

static esp_err_t load_credentials(credentials_t *credentials)
{
    if (!credentials) {
        return ESP_ERR_INVALID_ARG;
    }

    nvs_handle_t handle;
    esp_err_t error = nvs_open(NVS_NAMESPACE, NVS_READONLY, &handle);
    if (error != ESP_OK) {
        return error;
    }

    size_t ssid_length = sizeof(credentials->ssid);
    size_t password_length = sizeof(credentials->password);
    error = nvs_get_str(handle, NVS_SSID_KEY, credentials->ssid, &ssid_length);
    if (error == ESP_OK) {
        error = nvs_get_str(handle, NVS_PASSWORD_KEY, credentials->password, &password_length);
    }
    nvs_close(handle);
    return error;
}

static esp_err_t erase_credentials(void)
{
    nvs_handle_t handle;
    esp_err_t error = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        return ESP_OK;
    }
    if (error != ESP_OK) {
        return error;
    }

    const esp_err_t ssid_error = nvs_erase_key(handle, NVS_SSID_KEY);
    const esp_err_t password_error = nvs_erase_key(handle, NVS_PASSWORD_KEY);
    if ((ssid_error != ESP_OK && ssid_error != ESP_ERR_NVS_NOT_FOUND) ||
        (password_error != ESP_OK && password_error != ESP_ERR_NVS_NOT_FOUND)) {
        error = ssid_error != ESP_OK && ssid_error != ESP_ERR_NVS_NOT_FOUND ?
            ssid_error : password_error;
    } else {
        error = nvs_commit(handle);
    }
    nvs_close(handle);
    return error;
}

static esp_err_t read_console_line(const char *prompt, char *output,
                                   size_t capacity, bool hidden)
{
    if (!prompt || !output || capacity < 2) {
        return ESP_ERR_INVALID_ARG;
    }

    printf("%s", prompt);
    fflush(stdout);
    size_t length = 0;
    bool overflow = false;
    while (true) {
        uint8_t character = 0;
        const int received = uart_read_bytes(UART_NUM_0, &character, 1, portMAX_DELAY);
        if (received != 1) {
            return ESP_FAIL;
        }
        if (character == '\r' || character == '\n') {
            printf("\r\n");
            break;
        }
        if (character == 3) {
            printf("\r\n");
            return ESP_ERR_INVALID_STATE;
        }
        if (character == 8 || character == 127) {
            if (length > 0) {
                output[--length] = '\0';
                if (!hidden) {
                    uart_write_bytes(UART_NUM_0, "\b \b", 3);
                }
            }
            continue;
        }
        if (character < 0x20 || character == 0x7f) {
            continue;
        }
        if (length + 1 >= capacity) {
            overflow = true;
            continue;
        }
        output[length++] = static_cast<char>(character);
        output[length] = '\0';
        if (!hidden) {
            uart_write_bytes(UART_NUM_0, reinterpret_cast<const char *>(&character), 1);
        }
    }

    return overflow ? ESP_ERR_INVALID_SIZE : ESP_OK;
}

static bool valid_password(const char *password)
{
    const size_t length = strlen(password);
    if (length == 0 || (length >= 8 && length <= 63)) {
        return true;
    }
    if (length != 64) {
        return false;
    }
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(password); *p; ++p) {
        if (!isxdigit(*p)) {
            return false;
        }
    }
    return true;
}

static void generate_transfer_token(void)
{
    static constexpr char HEX[] = "0123456789abcdef";
    uint8_t random_bytes[TOKEN_BYTES] = {};
    esp_fill_random(random_bytes, sizeof(random_bytes));
    for (size_t i = 0; i < sizeof(random_bytes); ++i) {
        s_transfer_token[i * 2] = HEX[random_bytes[i] >> 4];
        s_transfer_token[i * 2 + 1] = HEX[random_bytes[i] & 0x0f];
    }
    s_transfer_token[TOKEN_HEX_LENGTH] = '\0';
}

static int cmd_wifi(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "help") == 0) {
        printf("wifi setup | status | connect | disconnect | forget | token\n");
        return 0;
    }

    if (strcmp(argv[1], "setup") == 0) {
        credentials_t credentials = {};
        esp_err_t error = read_console_line(
            "SSID: ", credentials.ssid, sizeof(credentials.ssid), false);
        if (error != ESP_OK) {
            printf("SSID input failed: %s\n", esp_err_to_name(error));
            return 1;
        }
        if (credentials.ssid[0] == '\0') {
            printf("SSID must not be empty.\n");
            return 1;
        }

        error = read_console_line(
            "Wi-Fi password (hidden; empty for an open network): ",
            credentials.password, sizeof(credentials.password), true);
        if (error != ESP_OK) {
            printf("Password input failed: %s\n", esp_err_to_name(error));
            return 1;
        }
        if (!valid_password(credentials.password)) {
            printf("Password must be empty, 8-63 characters, or a 64-digit hexadecimal PSK.\n");
            return 1;
        }

        error = save_credentials(credentials);
        if (error != ESP_OK) {
            printf("Cannot save Wi-Fi credentials to NVS: %s\n", esp_err_to_name(error));
            return 1;
        }
        error = start_station(credentials);
        if (error != ESP_OK) {
            printf("Credentials are saved, but Wi-Fi could not start: %s\n",
                   esp_err_to_name(error));
            return 1;
        }
        printf("Credentials saved in NVS; connection is starting. Use 'wifi status'.\n");
        return 0;
    }

    if (strcmp(argv[1], "connect") == 0) {
        credentials_t credentials = {};
        const esp_err_t error = load_credentials(&credentials);
        if (error != ESP_OK) {
            if (error == ESP_ERR_NVS_NOT_FOUND) {
                printf("No saved Wi-Fi credentials. Run 'wifi setup'.\n");
            } else {
                printf("Cannot read Wi-Fi credentials: %s\n", esp_err_to_name(error));
            }
            return 1;
        }
        const esp_err_t start_error = start_station(credentials);
        if (start_error != ESP_OK) {
            printf("Wi-Fi could not start: %s\n", esp_err_to_name(start_error));
            return 1;
        }
        printf("Wi-Fi connection is starting. Use 'wifi status'.\n");
        return 0;
    }

    if (strcmp(argv[1], "disconnect") == 0) {
        const esp_err_t error = stop_station();
        if (error != ESP_OK) {
            printf("Wi-Fi stop failed: %s\n", esp_err_to_name(error));
            return 1;
        }
        printf("Wi-Fi disconnected; saved credentials are unchanged.\n");
        return 0;
    }

    if (strcmp(argv[1], "forget") == 0) {
        esp_err_t error = stop_station();
        if (error != ESP_OK) {
            printf("Wi-Fi stop failed: %s\n", esp_err_to_name(error));
            return 1;
        }
        error = erase_credentials();
        if (error != ESP_OK) {
            printf("Cannot erase saved Wi-Fi credentials: %s\n", esp_err_to_name(error));
            return 1;
        }
        printf("Saved Wi-Fi credentials erased from NVS.\n");
        return 0;
    }

    if (strcmp(argv[1], "status") == 0) {
        credentials_t credentials = {};
        const esp_err_t credentials_error = load_credentials(&credentials);
        if (credentials_error == ESP_OK) {
            printf("Saved SSID: %s\n", credentials.ssid);
            printf("Credential storage: NVS (not flash-encrypted by this example).\n");
        } else if (credentials_error == ESP_ERR_NVS_NOT_FOUND) {
            printf("No saved Wi-Fi credentials.\n");
        } else {
            printf("Cannot read Wi-Fi credential status: %s\n",
                   esp_err_to_name(credentials_error));
            return 1;
        }

        if (s_connected.load() && s_sta_netif) {
            esp_netif_ip_info_t ip_info = {};
            const esp_err_t error = esp_netif_get_ip_info(s_sta_netif, &ip_info);
            if (error != ESP_OK) {
                printf("Connected, but IP address is unavailable: %s\n",
                       esp_err_to_name(error));
                return 1;
            }
            if (s_sd_available) {
                printf("Connected. Transfer URL: http://" IPSTR ":8080/api/{models,audio,results}/<filename>\n",
                       IP2STR(&ip_info.ip));
                printf("Transfer token: run 'wifi token' (required in Authorization header).\n");
            } else {
                printf("Connected, but file transfer is unavailable because the SD card is not mounted.\n");
            }
        } else {
            printf("Not connected. Use 'wifi connect' or 'wifi setup'.\n");
        }
        return 0;
    }

    if (strcmp(argv[1], "token") == 0) {
        printf("Transfer token: %s\n", s_transfer_token);
        return 0;
    }

    printf("Unknown wifi subcommand. Run 'wifi help'.\n");
    return 1;
}

} // namespace

esp_err_t wifi_transfer_init(void)
{
    s_server_mutex = xSemaphoreCreateMutex();
    if (!s_server_mutex) {
        return ESP_ERR_NO_MEM;
    }
    generate_transfer_token();

    struct stat status = {};
    s_sd_available = stat("/sd", &status) == 0 && S_ISDIR(status.st_mode);
    if (s_sd_available) {
        const char *directories[] = {"/sd/models", "/sd/audio", "/sd/results"};
        for (const char *directory : directories) {
            if (mkdir(directory, 0775) != 0 && errno != EEXIST) {
                ESP_LOGE(TAG, "Cannot create SD directory %s: errno=%d", directory, errno);
                return ESP_FAIL;
            }
        }
    } else {
        ESP_LOGW(TAG, "SD card is not mounted; WLAN file transfer will be unavailable");
    }

    credentials_t credentials = {};
    const esp_err_t error = load_credentials(&credentials);
    if (error == ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGI(TAG, "No saved WLAN credentials; configure with 'wifi setup'");
        return ESP_OK;
    }
    if (error != ESP_OK) {
        ESP_LOGE(TAG, "Cannot read WLAN configuration from NVS: %s", esp_err_to_name(error));
        return error;
    }

    ESP_LOGI(TAG, "Found saved WLAN configuration; starting station without logging credentials");
    const esp_err_t start_error = start_station(credentials);
    if (start_error != ESP_OK) {
        ESP_LOGE(TAG, "Cannot start WLAN station: %s", esp_err_to_name(start_error));
    }
    return start_error;
}

void wifi_transfer_register_commands(void)
{
    const esp_console_cmd_t command = {
        .command = "wifi",
        .help = "Configure WLAN and SD-card file transfer",
        .hint = "<setup|status|connect|disconnect|forget|token>",
        .func = &cmd_wifi,
        .argtable = nullptr,
        .func_w_context = nullptr,
        .context = nullptr
    };
    ESP_ERROR_CHECK(esp_console_cmd_register(&command));
}
