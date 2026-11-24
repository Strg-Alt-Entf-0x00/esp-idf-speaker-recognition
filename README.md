# Speaker Recognition for ESP-IDF

This component provides speaker recognition and verification for ESP32-P4
applications. It extracts L2-normalized speaker embeddings from 16 kHz mono
audio and performs 1:1 speaker verification and 1:N speaker identification
with persistent non-volatile storage (NVS).

The engine implements a dual-profile architecture, allowing applications to
select between maximum recognition accuracy and low-latency execution at
runtime:

- **High-Security Profile (`ECAPA-TDNN512`)**: Produces 192-dimensional
  embeddings using the WeSpeaker ECAPA-TDNN512 architecture. Designed for
  high-assurance access control, achieving an Equal Error Rate (EER) of ~0.8%.
- **Real-Time Profile (`TDNN-Tiny`)**: Produces 512-dimensional embeddings
  using a compact 5-layer TDNN architecture. Designed for low-latency
  voice control, executing in ~142 ms on the ESP32-P4 (~198 ms end-to-end).

Audio feature extraction uses ESP-DSP (512-point real FFT and an 80-channel
triangular Mel filterbank). Neural network forward inference is accelerated
using ESP32-P4 RV32 PIE INT8 SIMD instructions and dual-core FreeRTOS SMP
parallelization.

---

## Supported Configuration

- **Target**: ESP32-P4 (revision 0.1 or later)
- **ESP-IDF**: 6.0.2 or later
- **Audio Input**: 16 kHz, 16-bit signed mono PCM
- **Input Duration**: Dynamic; 1.5 s to 3.0 s recommended for optimal separation
- **Memory**: External PSRAM (Hex-SPI mode, 200 MHz recommended)

---

## Model Profiles

| Parameter | High-Security Profile | Real-Time Profile |
| :--- | :--- | :--- |
| **Architecture** | WeSpeaker ECAPA-TDNN512 | TDNN-Tiny |
| **Embedding Dimension** | 192-D (unit L2 vector) | 512-D (unit L2 vector) |
| **Equal Error Rate (EER)** | ~0.80% (FP16) / ~0.82% (INT8) | ~1.38% (FP16) / ~1.40% (INT8) |
| **Separation Margin** | 0.2796 (genuine 0.397 vs. impostor 0.117) | Tight hyperspherical cone |
| **Default Threshold** | 0.300 | 0.985 |
| **Inference Time (INT8)** | ~2,648 ms | ~142 ms |
| **End-to-End Latency** | ~2,700 ms | ~198 ms |
| **Model Size (INT8)** | 6.05 MB (`ecapa_tdnn512_int8.espm`) | 1.79 MB (`tdnn_tiny_int8.espm`) |
| **Model Size (FP16)** | 11.88 MB (`ecapa_tdnn512_fp16.espm`) | 3.54 MB (`tdnn_tiny_fp16.espm`) |
| **Primary Use Case** | Biometric access, door locks, admin login | Voice assistants, smart-home commands |

Profiles can be switched dynamically at runtime using
`esp_speaker_recognition_set_profile()` without restarting the device.

---

## Adding to an ESP-IDF Project

### 1. Component Dependency

Add the component to your project's `main/idf_component.yml`:

```yaml
dependencies:
  esp-idf-speaker-recognition:
    override_path: "../../esp-idf-speaker-recognition"
```

Or add the component directory to `EXTRA_COMPONENT_DIRS` in your project's root `CMakeLists.txt`:

```cmake
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/components/esp-idf-speaker-recognition")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
```

### 2. Kconfig Configuration (`sdkconfig.defaults`)

Ensure the following options are configured for the ESP32-P4:

```ini
CONFIG_IDF_TARGET="esp32p4"
CONFIG_SPIRAM=y
CONFIG_SPIRAM_SPEED_200M=y
CONFIG_SPIRAM_MODE_HEX=y
CONFIG_SPI_FLASH_AUTO_SUSPEND=y
CONFIG_EH_HOST_PORT_DMA_PREFER_SPIRAM=y
CONFIG_ESP_SYSTEM_EVENT_TASK_STACK_SIZE=4096
```

---

## Model Storage & Flash Partitions

The dual-profile runtime loads model containers from either dedicated SPI Flash
partitions or directly from a filesystem (such as a MicroSD card or LittleFS).

### Option A: SPI Flash Partitions

Configure your project's `partitions.csv` with dedicated entries for the models:

```csv
# Name,     Type, SubType, Offset,   Size,     Flags
nvs,        data, nvs,     0x9000,   0x6000,
phy_init,   data, phy,     0xf000,   0x1000,
factory,    app,  factory, 0x10000,  0x400000,
model,      data, 0x50,    0x500000, 0x800000, # 8 MB High-Security (ECAPA-TDNN512)
model_rt,   data, 0x51,    0xd00000, 0x400000, # 4 MB Real-Time (TDNN-Tiny)
```

Flash the compiled models using `esptool`:

```bash
python -m esptool --chip esp32p4 -p PORT write_flash \
    0x500000 models/espm/ecapa_tdnn512_int8.espm \
    0xd00000 models/espm/tdnn_tiny_int8.espm
```

### Option B: MicroSD Card / Filesystem

Alternatively, copy the `.espm` files onto a MicroSD card (e.g. `/sd/models/`):

```c
// Initialize directly from an SD card or FATFS file:
esp_err_t ret = esp_speaker_recognition_init_file("/sd/models/ecapa_tdnn512_int8.espm", 0.30f);
```

---

## Basic Usage

```c
#include "esp_speaker_recognition.h"
#include "esp_log.h"

static const char *TAG = "APP";

void app_main(void) {
    // 1. Initialize speaker recognition with the default High-Security profile
    esp_err_t ret = esp_speaker_recognition_init_partition("model", 0.30f);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Initialization failed: %s", esp_err_to_name(ret));
        return;
    }

    // 2. Enroll a speaker from 16 kHz PCM audio (signed 16-bit mono)
    ret = esp_speaker_recognition_enroll("alice", pcm_samples, num_samples);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Speaker 'alice' enrolled successfully");
    }

    // 3. 1:1 Verification against an enrolled speaker
    esp_sr_verify_result_t verify_res;
    ret = esp_speaker_recognition_verify("alice", probe_samples, probe_len, &verify_res);
    if (ret == ESP_OK && verify_res.verified) {
        ESP_LOGI(TAG, "Verified: %s (similarity: %.3f)",
                 verify_res.matched_name, verify_res.similarity_score);
    }

    // 4. 1:N Identification against all enrolled speakers in NVS
    esp_sr_identify_result_t id_res;
    ret = esp_speaker_recognition_identify(probe_samples, probe_len, &id_res);
    if (ret == ESP_OK && id_res.identified) {
        ESP_LOGI(TAG, "Identified speaker: %s (similarity: %.3f)",
                 id_res.matched_name, id_res.similarity_score);
    }

    // 5. Switch to Real-Time Profile for low-latency voice commands
    ret = esp_speaker_recognition_set_profile(ESP_SR_PROFILE_REALTIME);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Switched to Real-Time profile");
    }
}
```

---

## Model Format (ESPM) & Tooling

Models are stored in the **ESPM** (Embedded Secure Packed Model) container
format. The container is specifically engineered for ESP32-P4 RISC-V SIMD:

- Header validation with magic bytes and format version
- Deterministic CRC32 checksums for each section and tensor
- 64-byte aligned tensor offsets matching ESP32-P4 cache lines and DMA
- Symmetric per-channel INT8 quantization and full-precision FP16 weights
- Architecture metadata (sample rate, embedding dimension, Mel bins)
- **Zero Third-Party ML Dependencies:** Pure PyTorch -> ESPM conversion (no ONNX, no ESP-DL runtime required)

The offline model tooling is located in [`tools-py/`](tools-py/):

```bash
# 1. Download upstream PyTorch model artifacts into models/py/
python tools-py/speaker_recognition_models_py_downloader/speaker_recognition_download_models_py.py

# 2. Convert PyTorch checkpoints to native ESPM containers in models/espm/
python tools-py/speaker_recognition_models_converter/speaker_recognition_convert_models.py --model all --precision int8
python tools-py/speaker_recognition_models_converter/speaker_recognition_convert_models.py --model all --precision fp16
```

On Windows, one-click batch scripts handle the end-to-end setup:
- [`tools-py/speaker_recognition_download_models_py.bat`](tools-py/speaker_recognition_download_models_py.bat): Downloads official WeSpeaker and Espressif upstream checkpoints into `models/py/`.
- [`tools-py/speaker_recognition_convert_models.bat`](tools-py/speaker_recognition_convert_models.bat): Exports all 4 production containers (`ecapa_tdnn512_int8.espm`, `tdnn_tiny_int8.espm`, `ecapa_tdnn512_fp16.espm`, `tdnn_tiny_fp16.espm`) into `models/espm/`.

---

## Hardware Validation & Benchmarks

The component was validated on a Waveshare ESP32-P4-WIFI6 development board
(Dual-Core RISC-V @ 360 MHz, 32 MB PSRAM, 32 MB Flash) with ESP-IDF 6.0.2
using real German speech samples:

| Metric | High-Security (`ECAPA-TDNN512`) | Real-Time (`TDNN-Tiny`) |
| :--- | :--- | :--- |
| **Embedding Dimension** | 192-D | 512-D |
| **Equal Error Rate (EER)** | ~0.8% | ~1.4% |
| **Audio Input Window** | 1.6 s – 1.9 s (159 – 188 frames) | 1.6 s – 1.9 s (159 – 188 frames) |
| **Audio Frontend (FFT + Mel)** | 48 ms | 48 ms |
| **Neural Network Inferenz** | 2,648 ms | 142 ms |
| **End-to-End Latency** | ~2,700 ms | 198 ms |
| **1:1 Cosine Matching Time** | < 1 µs | < 1 µs |
| **Profile Switch Time** | ~1,100 ms | ~580 ms |
| **PSRAM Memory Allocated** | 11.9 MB (working buffers & weights) | 1.8 MB (working buffers & weights) |
| **Internal SRAM Used** | ~95 KB | ~95 KB |

---

## Board Example

The [ESP32-P4 board example](examples/ws-esp32-p4-wifi6-console-sr-dev/README.md)
documents audio input (ES8311 onboard codec and optional INMP441 MEMS microphone),
flashing instructions, interactive console commands (`sr_profile`, `sr_enroll`,
`sr_verify`, `sr_bench`), and Wi-Fi file transfer.

---

## Public API Reference

| Function | Description |
| :--- | :--- |
| `esp_speaker_recognition_init_partition` | Initialize engine from a named Flash partition. |
| `esp_speaker_recognition_init_file` | Initialize engine from a filesystem `.espm` file. |
| `esp_speaker_recognition_get_profile` | Get current active model profile (`ESP_SR_PROFILE_HIGH_SECURITY` or `ESP_SR_PROFILE_REALTIME`). |
| `esp_speaker_recognition_set_profile` | Dynamically switch active profile and reload model without rebooting. |
| `esp_speaker_recognition_extract_embedding`| Extract normalized speaker embedding vector (192-D or 512-D). |
| `esp_speaker_recognition_compute_similarity`| Compute cosine similarity between two embeddings. |
| `esp_speaker_recognition_enroll` | Extract embedding and register identity in persistent NVS. |
| `esp_speaker_recognition_enroll_embedding` | Store precomputed embedding directly into NVS. |
| `esp_speaker_recognition_verify` | Perform 1:1 verification of audio against a named profile. |
| `esp_speaker_recognition_identify` | Perform 1:N identification against all enrolled profiles. |
| `esp_speaker_recognition_delete` | Remove a speaker identity from database and NVS. |
| `esp_speaker_recognition_clear_all` | Remove all profiles from database and erase NVS records. |
| `esp_speaker_recognition_get_count` | Return count of currently registered speakers. |
| `esp_speaker_recognition_get_names` | Retrieve list of registered speaker identifiers. |
| `esp_speaker_recognition_get_threshold` | Query active cosine similarity verification threshold. |
| `esp_speaker_recognition_set_threshold` | Update verification threshold (range: 0.0 to 1.0). |
| `esp_speaker_recognition_get_embedding_dim`| Return active embedding dimension (192 or 512). |
| `esp_speaker_recognition_deinit` | Release all allocated buffers, frontend resources, and state. |

---

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Free for noncommercial use**: Personal, educational, research, and hobbyist projects are completely free.
- **Commercial use**: Not permitted without prior authorization. For commercial inquiries, please contact the author via GitHub.
