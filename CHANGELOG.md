# Changelog

All notable changes to `esp-idf-speaker-recognition` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-11-23

### Added
- **Dual-Profile Architecture**:
  - **High-Security Profile (`ESP_SR_PROFILE_HIGH_SECURITY`)**: WeSpeaker ECAPA-TDNN512 (192-D embedding, 0.8% EER, ~2.65 s latency) for access control and biometric authentication.
  - **Real-Time Profile (`ESP_SR_PROFILE_REALTIME`)**: TDNN-Tiny (512-D embedding, 142 ms raw inference, 198 ms end-to-end, 1.4% EER) for low-latency voice command interfaces.
  - Dynamic runtime profile switching via `esp_speaker_recognition_set_profile()` and console command `sr_profile <security|realtime>` in ~500 ms without reboot.
  - Profile-aware NVS storage supporting variable embedding dimensions (192-D and 512-D).
- **Hardware RV32 PIE INT8 SIMD Kernel (`sr_dotprod_pie.S`)**:
  - Handcrafted RISC-V assembly using ESP32-P4 PIE extension instructions (`esp.zero.xacc`, `esp.vld.128.ip`, `esp.vmulas.s8.xacc.ld.ip`, `esp.srs.s.xacc`).
  - Computes 4 MACs per cycle with 40-bit internal accumulation and automatic 16-byte alignment detection with scalar fallback.
- **General 1D Convolution Dispatcher with L1 SRAM Caching (`sr_simd_pie.c`)**:
  - Dual-Core FreeRTOS SMP parallelization across CPU0 and CPU1 via `sr_parallel_run()`.
  - Fast-path 1x1 INT8 convolution and multi-tap ($K \ge 1$, dilation $\ge 1$) convolution with L1 SRAM weight caching.
  - Reduced TDNN-Tiny convolution layer latency from 738 ms down to 142 ms.
- **Dual Partitioning Setup & ESPM Export**:
  - Partition `model` (subtype `0x50` @ `0x500000`, 8 MB): Holds `ecapa_tdnn512_int8.espm`.
  - Partition `model_rt` (subtype `0x51` @ `0xd00000`, 4 MB): Holds `tdnn_tiny_int8.espm`.
  - `tools-py/export_tdnn_tiny_to_espm.py`: Converts TDNN-Tiny into standard ESPM format.
- **PSRAM Buffer Allocation & FreeRTOS Stability**:
  - Configured ESP-Hosted Wi-Fi SDIO DMA to use PSRAM (`CONFIG_EH_HOST_PORT_DMA_PREFER_SPIRAM=y`).
  - Increased `console_repl` task stack to 16 KB and allocated intermediate layer buffers (`tdnn_stats`, `tdnn_seg1`, `tdnn_raw_emb`) from PSRAM heap.

---

## [1.0.0] - 2026-11-23

### Added
- **WeSpeaker ECAPA-TDNN512 Model Support**:
  - 1D dilated convolutional neural network forward engine.
  - Res2Net blocks with multi-scale feature extraction (scales = 4).
  - Squeeze-and-Excitation (SE) channel-attention modules.
  - Multi-layer feature aggregation (layers 2, 3, and 4).
  - Attentive Statistics Pooling (ASP) computing channel-dependent weighted mean and standard deviation.
  - Linear fully-connected projection with L2 normalization producing 192-D speaker embeddings.
- **Audio Frontend**:
  - 512-point Real FFT powered by ESP-DSP.
  - 80-channel triangular Mel filterbank spanning 20 Hz to 8000 Hz.
  - Real-time Cepstral Mean Normalization (CMN) with pre-emphasis ($0.97$).
- **ESPM Binary Model Format**:
  - Header validation with CRC32 checksums, version checks, and 64-byte alignment.
  - Support for INT8 per-channel and FP16 quantization.
  - Extensible key-value metadata container for sample rates, FFT sizes, and architecture properties.
- **Streaming Inference & Weight Caching**:
  - Ping-pong activation memory manager supporting residual additions without memory copies.
  - PSRAM weight caching for attentive pooling layers.
  - Sub-microsecond cosine similarity matching via SIMD dot-product.
- **NVS-Backed Profile Management**:
  - Persistent storage for up to 16 enrolled speaker profiles.
  - 1:1 speaker verification with configurable similarity threshold (default `0.30f`).
  - 1:N speaker identification with cosine similarity matching.
- **Development CLI Console Application**:
  - Interactive shell (`examples/ws-esp32-p4-wifi6-console-sr-dev`) for ESP32-P4 with Wi-Fi and ES8311 & INMP441 audio drivers.
  - Commands: `info`, `mic`, `sr_enroll`, `sr_verify`, `sr_identify`, `sr_list`, `sr_delete`, `sr_clear`, `sr_bench`, and `wifi`.
