# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate private Prism quantization fixtures using its exported ggml CPU functions."""
import argparse
import ctypes
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="Prism libggml-base shared library")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data")
    args = parser.parse_args()
    library = ctypes.CDLL(args.library)
    rng = np.random.default_rng(20261013)
    for name, block_bytes in (("pq2_0", 34), ("ptq1_0", 28)):
        values = rng.normal(0, 0.4, (4, 256)).astype(np.float32)
        packed = np.empty((4 * 256 // 128 * block_bytes,), np.uint8)
        reference = np.empty_like(values)
        quantize = getattr(library, "quantize_row_" + name + "_ref")
        dequantize = getattr(library, "dequantize_row_" + name)
        for function in (quantize, dequantize):
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            function.restype = None
        quantize(values.ctypes.data, packed.ctypes.data, values.size)
        dequantize(packed.ctypes.data, reference.ctypes.data, reference.size)
        np.save(args.output / (name + "_qbytes.npy"), packed)
        np.save(args.output / (name + "_deq.npy"), reference.reshape(-1))


if __name__ == "__main__":
    main()
