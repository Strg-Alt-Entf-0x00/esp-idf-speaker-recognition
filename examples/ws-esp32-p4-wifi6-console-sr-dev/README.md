# ESP32-P4 Wi-Fi 6 Speaker Recognition Console Example

This example demonstrates speaker recognition on the Waveshare ESP32-P4-WIFI6
development board. It provides an interactive serial REPL console for live
audio capture, speaker enrollment, 1:1 verification, 1:N identification,
dynamic profile switching, automated hardware benchmarking, and Wi-Fi file
transfer.

Board-specific configuration and console commands are documented here; the
[root README](../../README.md) describes the reusable speaker recognition
component.

---

## Hardware Requirements

- **Development Board**: Waveshare ESP32-P4-WIFI6 (Rev 1.0 or later)
- **SoC**: ESP32-P4 RISC-V Dual-Core @ 360 MHz, 32 MB PSRAM, 32 MB Flash
- **Wi-Fi Co-processor**: ESP32-C6 connected via SDIO (ESP-Hosted transport)
- **Audio Inputs**:
  - Onboard ES8311 I2S audio codec with microphone input (default)
  - Optional external INMP441 I2S digital MEMS microphone
- **Serial Connection**: USB-to-UART or Native USB-JTAG port at 115200 baud

### Optional INMP441 Microphone Pinout

When using an external INMP441 digital microphone, connect to the board's GPIO header:

| INMP441 Pin | ESP32-P4 Pin | Function |
| :--- | :--- | :--- |
| **VDD** | 3.3V | Power Supply |
| **GND** | GND | Ground |
| **SD** | GPIO 11 | Serial Data (DIN) |
| **WS** | GPIO 12 | Word Select (LRCK) |
| **SCK** | GPIO 13 | Bit Clock (BCLK) |
| **L/R** | GND | Left Channel Select |

Switch between the onboard ES8311 codec and the INMP441 at runtime using the `mic` command.

---

## Partition Layout

The example uses a custom partition table with dedicated entries for both model profiles:

```csv
# Name,     Type, SubType, Offset,   Size,     Flags
nvs,        data, nvs,     0x9000,   0x6000,
phy_init,   data, phy,     0xf000,   0x1000,
factory,    app,  factory, 0x10000,  0x400000,
model,      data, 0x50,    0x500000, 0x800000, # 8 MB High-Security (ECAPA-TDNN512)
model_rt,   data, 0x51,    0xd00000, 0x400000, # 4 MB Real-Time (TDNN-Tiny)
```

---

## Build and Flash

### 1. Build the Application

From this example directory:

```bash
idf.py build
```

### 2. Flash Firmware and Models

Flash the bootloader, partition table, application, and both ESPM model files:

```bash
# Flash application
idf.py -p PORT flash

# Flash High-Security Profile model (ECAPA-TDNN512)
python -m esptool --chip esp32p4 -p PORT write_flash 0x500000 ../../models/espm/ecapa_tdnn512_int8.espm

# Flash Real-Time Profile model (TDNN-Tiny)
python -m esptool --chip esp32p4 -p PORT write_flash 0xd00000 ../../models/espm/tdnn_tiny_int8.espm
```

Replace `PORT` with your serial device (e.g., `COM4` on Windows or `/dev/ttyUSB0` on Linux).

Alternatively, flash all binaries in a single `esptool` command:

```bash
python -m esptool --chip esp32p4 -p PORT -b 921600 write_flash \
    0x8000 build/partition_table/partition-table.bin \
    0x10000 build/ws-esp32-p4-wifi6-console-sr-dev.bin \
    0x500000 ../../models/espm/ecapa_tdnn512_int8.espm \
    0xd00000 ../../models/espm/tdnn_tiny_int8.espm
```

### 3. Open Serial Console

```bash
idf.py -p PORT monitor
```

---

## Interactive Console Commands

Once booted, the interactive prompt `esp32p4-sr>` is available:

```text
===================================================================
  ESP32-P4 NATIVE SPEAKER RECOGNITION CONSOLE
===================================================================
  Hardware : Waveshare ESP32-P4-WIFI6 (Dual-Core @ 360 MHz)
  PSRAM    : 10 / 32 MB free
  Internal : 95 / 430 KB free
  Commands : info, sr_profile, sr_enroll, sr_verify, sr_identify,
             sr_list, sr_delete, sr_clear, sr_bench, mic, wifi
===================================================================
```

### Command Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `info` | none | Display system status, active profile, threshold, and enrolled count |
| `sr_profile` | `<security\|realtime>` | Switch model profile dynamically without rebooting |
| `sr_enroll` | `<name> [-s <sec>] [-w <wav>]` | Enroll a new voice profile from microphone or WAV file |
| `sr_verify` | `<name> [-s <sec>] [-w <wav>]` | 1:1 verification of probe audio against an enrolled identity |
| `sr_identify`| `[-s <sec>] [-w <wav>]` | 1:N identification of probe audio against all enrolled profiles |
| `sr_list` | none | List all enrolled speaker profiles in NVS |
| `sr_delete` | `<name>` | Delete a speaker profile from NVS |
| `sr_clear` | none | Clear all enrolled speaker profiles |
| `sr_bench` | none | Run automated golden benchmark with German speech test samples |
| `mic` | `[status \| select <onboard\|inmp441> \| gain <0-42>]` | Query, select, or set gain for microphone input |
| `speaker_vol` | `<0-100>` | Set onboard speaker volume in percent |
| `play_wav` | `<file.wav>` | Play 16 kHz mono PCM16 WAV file through speaker |
| `record_mic` | `<output.wav> <seconds>` | Record from active microphone to a WAV file on SD card |
| `wifi` | `<setup\|connect\|status>` | Configure and manage Wi-Fi connection |

### Examples

#### Switch Model Profile
```text
esp32p4-sr> sr_profile realtime
Switched to Real-Time Profile (TDNN-Tiny 512-D, Sub-Second ~140ms)

esp32p4-sr> sr_profile security
Switched to High-Security Profile (ECAPA-TDNN512 192-D, 0.8% EER SOTA)
```

#### Enroll and Verify a Speaker
```text
# Enroll "alice" using 3 seconds of microphone recording
esp32p4-sr> sr_enroll alice -s 3
Recording 3 seconds from onboard microphone...
Extraction complete (192-D embedding). Enrolled 'alice' in NVS.

# Verify probe audio against "alice"
esp32p4-sr> sr_verify alice -s 3
Recording 3 seconds...
Cosine Similarity: 0.395 (Threshold: 0.300) -> VERIFIED (PASS)
```

#### Run Automated Hardware Benchmark
```text
esp32p4-sr> sr_bench
===================================================================
  RUNNING NATIVE SPEAKER RECOGNITION BENCHMARK (ESP32-P4)...
===================================================================
Step 1: Enrolling 'bench_alice' (1.90s German speech)...
Step 2: 1:1 Genuine Verification ('bench_alice')...
Step 3: 1:1 Impostor Verification (Unknown speaker)...
Step 4: 1:N Identification ('bench_bob')...

===================================================================
  SPEAKER RECOGNITION BENCHMARK SUMMARY
===================================================================
  Model Profile      : REALTIME (TDNN-Tiny Sub-Second)
  Embedding Output   : 512-D L2-Normalized Vector
  Inference Latency  : 142.2 ms NN (197.8 ms end-to-end)
  Cosine Similarity  :
    - Genuine Match  : 0.9943
    - Impostor Match : 0.9914
  Identification 1:N : Matched='bench_bob' (Score=1.0000, Status=PASSED)
  Hardware Status    : 100% Operational
===================================================================
```

---

## Wi-Fi Configuration

WLAN connectivity uses ESP-Hosted over SDIO with the onboard ESP32-C6 co-processor.

```text
# Configure network credentials (stored in NVS)
esp32p4-sr> wifi setup

# Connect to configured network
esp32p4-sr> wifi connect

# Check connection status and IP address
esp32p4-sr> wifi status
```
