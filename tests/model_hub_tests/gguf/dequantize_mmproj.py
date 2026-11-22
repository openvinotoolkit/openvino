# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Create an F32 copy of represented GGUF weights for llama.cpp CPU validation.

Use gguf-py from the pinned llama.cpp checkout. This preserves the quantized
checkpoint's represented weights; it does not recover the publisher's weights.
The pinned llama-quantize executable rejects the clip architecture.
"""
import argparse
from pathlib import Path

import gguf
import numpy as np


def dequantize_mmproj(source, destination):
    """Write an F32 copy of every tensor in source to destination, keeping all metadata."""
    reader = gguf.GGUFReader(source)
    writer = gguf.GGUFWriter(destination, reader.fields["general.architecture"].contents(), use_temp_file=True)
    if "general.alignment" in reader.fields:
        writer.add_custom_alignment(int(reader.fields["general.alignment"].contents()))
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.alignment"):
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             field.types[-1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, gguf.dequantize(tensor.data, tensor.tensor_type).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Source and destination must differ")
    dequantize_mmproj(args.source, args.destination)


if __name__ == "__main__":
    main()
