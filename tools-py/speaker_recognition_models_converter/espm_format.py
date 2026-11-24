"""ESPM v3 format definition and serialization for Speaker Recognition models.

ESPM (Espressif / Embedded Secure Packed Model) v3 container:
- Magic: 0x4D505345 ("ESPM")
- Format Version: 3
- Section Table: Metadata, Config JSON, Tensor Index, Tensor Data
- Deterministic CRC32 checksums for every tensor and section
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MAGIC = 0x4D505345  # 'ESPM'
HEADER_SIZE = 128
SECTION_ENTRY_SIZE = 40
ALIGNMENT = 64
V3_FORMAT_VERSION = 3
V3_MAX_SECTIONS = 16
V3_INDEX_HEADER_SIZE = 36
V3_TENSOR_RECORD_SIZE = 160
V3_MAX_TENSORS = 512
V3_SECTION_OPTIONAL = 1

V3_METADATA_HEADER = struct.Struct("<4sHHII")
V3_METADATA_RECORD = struct.Struct("<HBBI")
V3_INDEX_HEADER = struct.Struct("<4sHHHHIIQQ")
V3_TENSOR = struct.Struct("<II4BI8IIB3sQIQ Q64s12s")

assert V3_METADATA_HEADER.size == 16
assert V3_METADATA_RECORD.size == 8
assert V3_INDEX_HEADER.size == V3_INDEX_HEADER_SIZE
assert V3_TENSOR.size == V3_TENSOR_RECORD_SIZE

# Model and Quantization types
MODEL_TYPE_SPEAKER_RECOGNITION = 4

MODEL_QUANT_INT8 = 1
MODEL_QUANT_INT8_PER_CH = 2
MODEL_QUANT_FP16 = 3
MODEL_QUANT_FP32 = 4

# Data types
DTYPE_INT8 = 1
DTYPE_INT16 = 2
DTYPE_FP16 = 3
DTYPE_FP32 = 4

# Section IDs
SECTION_CONFIG_JSON = 5
SECTION_TENSOR_INDEX = 6
SECTION_TENSOR_DATA = 7
SECTION_METADATA = 8

_HEADER = struct.Struct("<IHHHH32s16sIIIQIHHIII28s")
_SECTION = struct.Struct("<HHQQI16s")
assert _HEADER.size == HEADER_SIZE
assert _SECTION.size == SECTION_ENTRY_SIZE


def align_offset(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def fixed_string(value: str, capacity: int, field_name: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) >= capacity or b"\0" in encoded:
        raise ValueError(f"{field_name} must fit in {capacity - 1} UTF-8 bytes")
    return encoded + bytes(capacity - len(encoded))


def encode_v3_metadata(records: Dict[str, Tuple[int, bytes]]) -> bytes:
    encoded = bytearray(V3_METADATA_HEADER.pack(b"EMET", 1, 16, len(records), 0))
    for key in sorted(records):
        key_bytes = key.encode("utf-8")
        value_type, value = records[key]
        if not key_bytes or b"\0" in key_bytes or len(key_bytes) > 0xFFFF:
            raise ValueError(f"invalid metadata key {key!r}")
        encoded.extend(V3_METADATA_RECORD.pack(len(key_bytes), value_type, 1, len(value)))
        encoded.extend(key_bytes)
        encoded.extend(value)
    return bytes(encoded)


def meta_u32(val: int) -> Tuple[int, bytes]:
    return 1, struct.pack("<I", int(val))


def meta_f32(val: float) -> Tuple[int, bytes]:
    return 4, struct.pack("<f", float(val))


def meta_utf8(val: str) -> Tuple[int, bytes]:
    return 6, val.encode("utf-8")


def write_speaker_espm(
    output_path: Path,
    *,
    model_name: str,
    version_str: str,
    quant_type: int,
    tensors: List[Dict[str, Any]],
    metadata: Dict[str, Tuple[int, bytes]],
    config_json: Optional[Dict[str, Any]] = None,
    sample_rate: int = 16000,
) -> Dict[str, Any]:
    """Write an ESPM v3 container for Speaker Recognition."""
    tensor_data = bytearray()
    tensor_records = []
    total_elements = 0
    total_params = 0
    total_data_bytes = 0

    for t in tensors:
        name = t["name"]
        shape = tuple(t["shape"])
        dtype = t.get("dtype", DTYPE_FP32)
        payload = t["data"]

        elements = math.prod(shape) if shape else 1
        tensor_data.extend(bytes((-len(tensor_data)) % 16))  # 16-byte alignment
        data_offset = len(tensor_data)
        tensor_data.extend(payload)

        dimensions = shape + (0,) * (8 - len(shape))
        name_bytes = fixed_string(name, 64, "tensor name")
        name_hash = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
        crc32_val = zlib.crc32(payload) & 0xFFFFFFFF

        # V3_TENSOR: name_hash, flags, dtype, ndim, quant_mode, reserved0, elements,
        # dims[8], part_id, reserved1, reserved2[3], crc32, reserved3, data_offset, data_size,
        # name[64], reserved4[12]
        tensor_records.append(
            V3_TENSOR.pack(
                name_hash,
                0,
                dtype,
                len(shape),
                0,
                0,
                elements,
                *dimensions,
                0,
                0xFF,
                bytes(3),
                crc32_val,
                0,
                data_offset,
                len(payload),
                name_bytes,
                bytes(12),
            )
        )
        total_elements += elements
        total_params += elements
        total_data_bytes += len(payload)

    # Encode sections
    encoded_metadata = encode_v3_metadata(metadata)
    encoded_index = V3_INDEX_HEADER.pack(
        b"ETIX",
        2,
        V3_INDEX_HEADER_SIZE,
        V3_TENSOR_RECORD_SIZE,
        0,
        len(tensor_records),
        1,
        total_elements,
        total_data_bytes,
    ) + b"".join(tensor_records)

    sections: List[Tuple[int, int, str, bytes]] = []
    if config_json:
        config_bytes = json.dumps(config_json, indent=2).encode("utf-8")
        sections.append((SECTION_CONFIG_JSON, V3_SECTION_OPTIONAL, "config_json", config_bytes))

    sections.append((SECTION_TENSOR_INDEX, 0, "tensor_index", encoded_index))
    sections.append((SECTION_TENSOR_DATA, 0, "tensor_data", bytes(tensor_data)))
    sections.append((SECTION_METADATA, 0, "metadata", encoded_metadata))

    if len(sections) > V3_MAX_SECTIONS:
        raise ValueError("Too many ESPM sections")

    table_end = HEADER_SIZE + len(sections) * SECTION_ENTRY_SIZE
    offset = align_offset(table_end)
    rows = []
    for kind, section_flags, sname, payload in sections:
        rows.append(
            (
                kind,
                section_flags,
                sname,
                offset,
                payload,
                zlib.crc32(payload) & 0xFFFFFFFF,
            )
        )
        offset = align_offset(offset + len(payload))

    file_size = rows[-1][3] + len(rows[-1][4])
    section_table = b"".join(
        _SECTION.pack(
            kind,
            section_flags,
            section_offset,
            len(payload),
            checksum,
            fixed_string(sname, 16, "section name"),
        )
        for kind, section_flags, sname, section_offset, payload, checksum in rows
    )
    table_crc = zlib.crc32(section_table) & 0xFFFFFFFF
    psram_required_kb = (len(tensor_data) + 1023) // 1024

    def pack_header(header_crc: int) -> bytes:
        return _HEADER.pack(
            MAGIC,
            V3_FORMAT_VERSION,
            MODEL_TYPE_SPEAKER_RECOGNITION,
            quant_type,
            len(rows),
            fixed_string(model_name, 32, "model name"),
            fixed_string(version_str, 16, "version"),
            sample_rate,
            psram_required_kb,
            header_crc,
            file_size,
            HEADER_SIZE,
            HEADER_SIZE,
            SECTION_ENTRY_SIZE,
            ALIGNMENT,
            0,
            table_crc,
            bytes(28),
        )

    header = pack_header(0)
    header = pack_header(zlib.crc32(header) & 0xFFFFFFFF)

    output = bytearray(file_size)
    output[:HEADER_SIZE] = header
    output[HEADER_SIZE : HEADER_SIZE + len(section_table)] = section_table
    for _, _, _, section_offset, payload, _ in rows:
        output[section_offset : section_offset + len(payload)] = payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)

    return {
        "file_size": file_size,
        "parameter_count": total_params,
        "weights_size": len(tensor_data),
        "model_name": model_name,
        "quant_type": quant_type,
        "tensors_count": len(tensors),
    }
