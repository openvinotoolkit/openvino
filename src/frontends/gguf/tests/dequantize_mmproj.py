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

from mmproj_fixtures import finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Source and destination must differ")
    reader = gguf.GGUFReader(args.source)
    writer = gguf.GGUFWriter(args.destination, reader.fields["general.architecture"].contents(), use_temp_file=True)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             field.types[-1] if len(field.types) > 1 else None)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, gguf.dequantize(tensor.data, tensor.tensor_type).astype(np.float32))
    finish(writer)


if __name__ == "__main__":
    main()
