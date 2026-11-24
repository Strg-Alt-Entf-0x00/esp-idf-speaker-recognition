#!/usr/bin/env python3
"""
Pure PyTorch to ESPM Container Exporter
======================================
Repository: esp-idf-speaker-recognition

Directly loads native PyTorch state_dict checkpoints (.pt / .pth) from models/py/
and exports production-ready ESPM binary containers to models/espm/.

Supported Architectures:
  1. WeSpeaker ECAPA-TDNN512 (High-Security Profile, 192-D)
  2. TDNN-Tiny (Real-Time Profile, 512-D)

Zero third-party ML runtime dependencies (No ONNX, No ESP-DL).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

from espm_format import (
    DTYPE_INT8,
    DTYPE_FP16,
    DTYPE_FP32,
    MODEL_QUANT_INT8_PER_CH,
    MODEL_QUANT_FP16,
    MODEL_QUANT_FP32,
    meta_u32,
    meta_utf8,
    write_speaker_espm,
)


def compute_bn_scale_bias(gamma: np.ndarray, beta: np.ndarray, mean: np.ndarray, var: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    scale = (gamma / np.sqrt(var + eps)).astype(np.float32)
    bias = (beta - mean * scale).astype(np.float32)
    return scale, bias


def quantize_and_pack_tensor(tensors: List[Dict[str, Any]], name: str, arr: np.ndarray, is_weight: bool, precision: str):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    shape = list(arr.shape)

    if is_weight and precision == "int8":
        out_c = arr.shape[0]
        scales = np.zeros(out_c, dtype=np.float32)
        q_arr = np.zeros(arr.shape, dtype=np.int8)
        for c in range(out_c):
            max_v = float(np.max(np.abs(arr[c])))
            scale = max(max_v / 127.0, 1e-8)
            scales[c] = scale
            q_arr[c] = np.clip(np.round(arr[c] / scale), -127, 127).astype(np.int8)
        data = q_arr.tobytes()
        dtype = DTYPE_INT8
        tensors.append({
            "name": name + ".scale",
            "shape": [out_c],
            "dtype": DTYPE_FP32,
            "data": scales.tobytes(),
        })
    elif is_weight and precision == "fp16":
        data = arr.astype(np.float16).tobytes()
        dtype = DTYPE_FP16
    else:
        data = arr.astype(np.float32).tobytes()
        dtype = DTYPE_FP32

    tensors.append({
        "name": name,
        "shape": shape,
        "dtype": dtype,
        "data": data,
    })


def export_ecapa(checkpoint_path: Path, output_path: Path, precision: str = "int8"):
    print(f"\n[ECAPA-TDNN512] Loading PyTorch checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    inits = {k: v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32) for k, v in sd.items()}
    print(f"Loaded {len(inits)} tensors from PyTorch state_dict.")

    def get_t(key: str) -> np.ndarray:
        if key in inits:
            return inits[key]
        if key.startswith("model.") and key[6:] in inits:
            return inits[key[6:]]
        if not key.startswith("model.") and f"model.{key}" in inits:
            return inits[f"model.{key}"]
        if key in ("mean_vec", "model.mean_vec"):
            return np.zeros(192, dtype=np.float32)
        raise KeyError(f"Tensor key '{key}' not found in checkpoint state_dict")

    tensors: List[Dict[str, Any]] = []
    quant_type = (
        MODEL_QUANT_INT8_PER_CH if precision == "int8"
        else MODEL_QUANT_FP16 if precision == "fp16"
        else MODEL_QUANT_FP32
    )

    def add_t(name: str, arr: np.ndarray, is_weight: bool = True):
        # Transpose Conv1D weights from PyTorch [out_c, in_c, K] to cache-linear [out_c, K, in_c] for RV32 PIE SIMD
        if is_weight and arr.ndim == 3 and arr.shape[2] > 1:
            arr = np.transpose(arr, (0, 2, 1))
        elif is_weight and arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        quantize_and_pack_tensor(tensors, name, arr, is_weight, precision)

    # 1. Layer 1: Conv1D (512, 80, 5) + BN (512)
    add_t("layer1.conv.weight", get_t("layer1.conv.weight"), is_weight=True)
    add_t("layer1.conv.bias", get_t("layer1.conv.bias"), is_weight=False)
    s, b = compute_bn_scale_bias(
        get_t("layer1.bn.weight"), get_t("layer1.bn.bias"),
        get_t("layer1.bn.running_mean"), get_t("layer1.bn.running_var")
    )
    add_t("layer1.bn.scale", s, is_weight=False)
    add_t("layer1.bn.bias", b, is_weight=False)

    # 2. Layers 2, 3, 4: SE-Res2Blocks
    for l_idx in range(2, 5):
        prefix = f"layer{l_idx}.se_res2block"
        out_prefix = f"layer{l_idx}"

        # conv0 (512, 512, 1) + bn0
        add_t(f"{out_prefix}.conv0.weight", get_t(f"{prefix}.0.conv.weight"), is_weight=True)
        add_t(f"{out_prefix}.conv0.bias", get_t(f"{prefix}.0.conv.bias"), is_weight=False)
        s, b = compute_bn_scale_bias(
            get_t(f"{prefix}.0.bn.weight"), get_t(f"{prefix}.0.bn.bias"),
            get_t(f"{prefix}.0.bn.running_mean"), get_t(f"{prefix}.0.bn.running_var")
        )
        add_t(f"{out_prefix}.conv0.bn_scale", s, is_weight=False)
        add_t(f"{out_prefix}.conv0.bn_bias", b, is_weight=False)

        # 7 branch convs (64, 64, 3) + bns
        for b_idx in range(7):
            add_t(f"{out_prefix}.branch{b_idx}.weight", get_t(f"{prefix}.1.convs.{b_idx}.weight"), is_weight=True)
            add_t(f"{out_prefix}.branch{b_idx}.bias", get_t(f"{prefix}.1.convs.{b_idx}.bias"), is_weight=False)
            s, b = compute_bn_scale_bias(
                get_t(f"{prefix}.1.bns.{b_idx}.weight"), get_t(f"{prefix}.1.bns.{b_idx}.bias"),
                get_t(f"{prefix}.1.bns.{b_idx}.running_mean"), get_t(f"{prefix}.1.bns.{b_idx}.running_var")
            )
            add_t(f"{out_prefix}.branch{b_idx}.bn_scale", s, is_weight=False)
            add_t(f"{out_prefix}.branch{b_idx}.bn_bias", b, is_weight=False)

        # conv2 (512, 512, 1) + bn2
        add_t(f"{out_prefix}.conv2.weight", get_t(f"{prefix}.2.conv.weight"), is_weight=True)
        add_t(f"{out_prefix}.conv2.bias", get_t(f"{prefix}.2.conv.bias"), is_weight=False)
        s, b = compute_bn_scale_bias(
            get_t(f"{prefix}.2.bn.weight"), get_t(f"{prefix}.2.bn.bias"),
            get_t(f"{prefix}.2.bn.running_mean"), get_t(f"{prefix}.2.bn.running_var")
        )
        add_t(f"{out_prefix}.conv2.bn_scale", s, is_weight=False)
        add_t(f"{out_prefix}.conv2.bn_bias", b, is_weight=False)

        # SE linear1 (128, 512) and linear2 (512, 128)
        add_t(f"{out_prefix}.se.linear1.weight", get_t(f"{prefix}.3.linear1.weight"), is_weight=True)
        add_t(f"{out_prefix}.se.linear1.bias", get_t(f"{prefix}.3.linear1.bias"), is_weight=False)
        add_t(f"{out_prefix}.se.linear2.weight", get_t(f"{prefix}.3.linear2.weight"), is_weight=True)
        add_t(f"{out_prefix}.se.linear2.bias", get_t(f"{prefix}.3.linear2.bias"), is_weight=False)

    # 3. Multi-layer Aggregation Conv (1536, 1536, 1)
    add_t("conv_agg.weight", get_t("conv.weight"), is_weight=True)
    add_t("conv_agg.bias", get_t("conv.bias"), is_weight=False)

    # 4. Attentive Statistics Pooling
    w_p1 = get_t("pool.linear1.weight")
    if w_p1.ndim == 3:
        w_p1 = w_p1.squeeze(-1)
    add_t("pool.linear1.weight", w_p1, is_weight=True)
    add_t("pool.linear1.bias", get_t("pool.linear1.bias"), is_weight=False)

    w_p2 = get_t("pool.linear2.weight")
    if w_p2.ndim == 3:
        w_p2 = w_p2.squeeze(-1)
    add_t("pool.linear2.weight", w_p2, is_weight=True)
    add_t("pool.linear2.bias", get_t("pool.linear2.bias"), is_weight=False)

    # 5. Output BN and Linear (192, 3072)
    s, b = compute_bn_scale_bias(
        get_t("bn.weight"), get_t("bn.bias"),
        get_t("bn.running_mean"), get_t("bn.running_var")
    )
    add_t("out.bn.scale", s, is_weight=False)
    add_t("out.bn.bias", b, is_weight=False)
    add_t("out.linear.weight", get_t("linear.weight"), is_weight=True)
    add_t("out.linear.bias", get_t("linear.bias"), is_weight=False)
    add_t("mean_vec", get_t("mean_vec"), is_weight=False)

    metadata = {
        "general.architecture": meta_utf8("wespeaker.ecapa_tdnn512"),
        "general.architecture_version": meta_u32(1),
        "speaker.sample_rate": meta_u32(16000),
        "speaker.embedding_dim": meta_u32(192),
        "speaker.mel_bins": meta_u32(80),
        "speaker.frame_length_ms": meta_u32(25),
        "speaker.frame_shift_ms": meta_u32(10),
        "speaker.fft_size": meta_u32(512),
        "speaker.precision": meta_utf8(precision),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    res = write_speaker_espm(
        output_path,
        model_name="ecapa_tdnn512",
        version_str="1.0.0",
        quant_type=quant_type,
        tensors=tensors,
        metadata=metadata,
        sample_rate=16000,
    )
    print(f"[OK] Wrote ECAPA-TDNN512 ({precision.upper()}): {output_path} ({res['file_size']/1024/1024:.2f} MB, {res['tensors_count']} tensors)")
    return res


def export_tdnn_tiny(checkpoint_path: Path, output_path: Path, precision: str = "int8"):
    print(f"\n[TDNN-Tiny] Loading PyTorch checkpoint: {checkpoint_path}")
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    inits = {k: v.detach().cpu().float().numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32) for k, v in sd.items()}
    print(f"Loaded {len(inits)} tensors from PyTorch state_dict.")

    tensors: List[Dict[str, Any]] = []
    quant_type = (
        MODEL_QUANT_INT8_PER_CH if precision == "int8"
        else MODEL_QUANT_FP16 if precision == "fp16"
        else MODEL_QUANT_FP32
    )

    def add_t(name: str, arr: np.ndarray, is_weight: bool = True):
        # In PyTorch: Conv1d weight is [out_c, in_c, K].
        # Transpose to [out_c, K, in_c] for RV32 PIE SIMD dot-product vectorization!
        if is_weight and arr.ndim == 3:
            arr = np.transpose(arr, (0, 2, 1))
        quantize_and_pack_tensor(tensors, name, arr, is_weight, precision)

    add_t("tdnn.conv1.weight", inits["tdnn.conv1.weight"], is_weight=True)
    add_t("tdnn.conv1.bias", inits["tdnn.conv1.bias"], is_weight=False)

    add_t("tdnn.conv2.weight", inits["tdnn.conv2.weight"], is_weight=True)
    add_t("tdnn.conv2.bias", inits["tdnn.conv2.bias"], is_weight=False)

    add_t("tdnn.conv3.weight", inits["tdnn.conv3.weight"], is_weight=True)
    add_t("tdnn.conv3.bias", inits["tdnn.conv3.bias"], is_weight=False)

    add_t("tdnn.conv4.weight", inits["tdnn.conv4.weight"], is_weight=True)
    add_t("tdnn.conv4.bias", inits["tdnn.conv4.bias"], is_weight=False)

    add_t("tdnn.conv5.weight", inits["tdnn.conv5.weight"], is_weight=True)
    add_t("tdnn.conv5.bias", inits["tdnn.conv5.bias"], is_weight=False)

    add_t("tdnn.fc1.weight", inits["tdnn.fc1.weight"], is_weight=True)
    add_t("tdnn.fc1.bias", inits["tdnn.fc1.bias"], is_weight=False)

    add_t("tdnn.fc2.weight", inits["tdnn.fc2.weight"], is_weight=True)
    add_t("tdnn.fc2.bias", inits["tdnn.fc2.bias"], is_weight=False)

    metadata = {
        "model_arch": meta_u32(3),         # SR_ARCH_TDNN_TINY
        "embedding_dim": meta_u32(512),    # 512-D
        "mel_bins": meta_u32(80),          # 80-bin Mel FBank
        "sample_rate": meta_u32(16000),    # 16 kHz mono
        "precision": meta_utf8(precision),
        "author": meta_utf8("Antigravity Dev Team"),
        "description": meta_utf8("TDNN-Tiny Real-Time Speaker Recognition Profile"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    res = write_speaker_espm(
        output_path=output_path,
        model_name="TDNN-Tiny-RealTime",
        version_str="1.0.0",
        quant_type=quant_type,
        tensors=tensors,
        metadata=metadata,
        sample_rate=16000,
    )
    print(f"[OK] Wrote TDNN-Tiny ({precision.upper()}): {output_path} ({res['file_size']/1024/1024:.2f} MB, {res['tensors_count']} tensors)")
    return res


def main():
    repo_root = Path(__file__).resolve().parents[2]
    py_dir = repo_root / "models" / "py"
    espm_dir = repo_root / "models" / "espm"

    parser = argparse.ArgumentParser(description="Export pure PyTorch checkpoints (.pt / .pth) to native ESPM format")
    parser.add_argument("--model", choices=["ecapa", "tdnn_tiny", "all"], default="all", help="Model to export (default: all)")
    parser.add_argument("--precision", choices=["int8", "fp16", "fp32"], default="int8", help="Target precision (default: int8)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Custom checkpoint path (optional)")
    parser.add_argument("--output", type=Path, default=None, help="Custom output path (optional)")
    args = parser.parse_args()

    if args.model in ("ecapa", "all"):
        ckpt = args.checkpoint or (py_dir / "ecapa_tdnn512.pt")
        out = args.output or (espm_dir / f"ecapa_tdnn512_{args.precision}.espm")
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt}")
            return 1
        export_ecapa(ckpt, out, args.precision)

    if args.model in ("tdnn_tiny", "all"):
        ckpt = args.checkpoint or (py_dir / "tdnn_tiny.pth")
        out = args.output or (espm_dir / f"tdnn_tiny_{args.precision}.espm")
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {ckpt}")
            return 1
        export_tdnn_tiny(ckpt, out, args.precision)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
