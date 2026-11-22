# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write small synthetic GGUF inputs for the extension examples (standard library only)."""

import argparse
import math
import struct
from pathlib import Path


def string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def write(path, metadata, tensors):
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += string(key)
        if isinstance(value, bool):
            header += struct.pack("<I?", 7, value)
        elif isinstance(value, str):
            header += struct.pack("<I", 8) + string(value)
        elif isinstance(value, float):
            header += struct.pack("<If", 6, value)
        else:
            header += struct.pack("<II", 4, value)
    payload = bytearray()
    for name, dimensions, values in tensors:
        payload.extend(bytes((-len(payload)) % 32))
        header += string(name) + struct.pack("<I", len(dimensions))
        header += struct.pack("<" + "Q" * len(dimensions), *dimensions)
        header += struct.pack("<IQ", 0, len(payload))
        payload.extend(struct.pack("<" + "f" * len(values), *values))
    path.write_bytes(header + bytes((-len(header)) % 32) + payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    write(args.output / "projection.gguf", {"general.architecture": "example-projector"},
          [("projection.weight", [2, 3], weights)])
    write(args.output / "mmproj.gguf", {
        "general.architecture": "clip",
        "clip.has_audio_encoder": True,
        "clip.audio.projector_type": "example-linear",
    }, [("example.audio.projection.weight", [2, 3], weights)])

    arch = "example-qwen3"
    metadata = {
        "general.architecture": arch,
        f"{arch}.block_count": 1,
        f"{arch}.embedding_length": 8,
        f"{arch}.attention.head_count": 2,
        f"{arch}.attention.head_count_kv": 1,
        f"{arch}.rope.dimension_count": 4,
        f"{arch}.context_length": 32,
        f"{arch}.attention.layer_norm_rms_epsilon": 1e-5,
    }
    tensors = []

    def weight(name, dimensions, norm=False):
        count = math.prod(dimensions)
        values = [1.0 + 0.01 * math.sin(i) if norm else 0.05 * math.sin(i + 1) for i in range(count)]
        tensors.append((name, dimensions, values))

    weight("token_embd.weight", [8, 16])
    weight("output_norm.weight", [8], True)
    weight("blk.0.attn_norm.weight", [8], True)
    for name, dimensions in [("attn_q", [8, 8]), ("attn_k", [8, 4]), ("attn_v", [8, 4]),
                             ("attn_output", [8, 8])]:
        weight(f"blk.0.{name}.weight", dimensions)
    for name in ["attn_q_norm", "attn_k_norm"]:
        weight(f"blk.0.{name}.weight", [4], True)
    weight("blk.0.ffn_norm.weight", [8], True)
    weight("blk.0.ffn_gate.weight", [8, 12])
    weight("blk.0.ffn_up.weight", [8, 12])
    weight("blk.0.ffn_down.weight", [12, 8])
    write(args.output / "decoder.gguf", metadata, tensors)
    print(f"Wrote projection.gguf, mmproj.gguf and decoder.gguf to {args.output}")


if __name__ == "__main__":
    main()
