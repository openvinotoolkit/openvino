# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate Q8_1 fixtures with pinned ggml and its Q8_0 dequantization oracle."""

import argparse
import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


LLAMA_CPP_COMMIT = "476c01efe88aad7880a8132d5d3a415f2ca75139"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cpp", type=Path, required=True, help="Pinned llama.cpp checkout")
    parser.add_argument("--library", type=Path, required=True, help="Its built libggml-base shared library")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != LLAMA_CPP_COMMIT:
        parser.error(f"llama.cpp must be at {LLAMA_CPP_COMMIT}, got {revision}")
    subprocess.run(
        ["git", "-C", str(args.llama_cpp), "diff", "--exit-code", LLAMA_CPP_COMMIT, "--", "ggml"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    library = ctypes.CDLL(str(args.library.resolve()))
    library.ggml_type_size.argtypes = [ctypes.c_int]
    library.ggml_type_size.restype = ctypes.c_size_t
    library.ggml_blck_size.argtypes = [ctypes.c_int]
    library.ggml_blck_size.restype = ctypes.c_int64
    if library.ggml_type_size(9) != 36 or library.ggml_blck_size(9) != 32:
        parser.error("The ggml library must use 36-byte, 32-element Q8_1 blocks")
    if library.ggml_type_size(8) != 34 or library.ggml_blck_size(8) != 32:
        parser.error("The ggml library must use 34-byte, 32-element Q8_0 blocks")

    quantize = library.quantize_row_q8_1_ref
    dequantize = library.dequantize_row_q8_0
    for function in (quantize, dequantize):
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        function.restype = None
    seed = 20261026
    values = (np.random.default_rng(seed).normal(0, 0.4, (4, 256)) + np.linspace(-0.2, 0.3, 256)).astype(
        np.float32
    )
    packed = np.empty((values.size // 32, 36), dtype=np.uint8)
    quantize(values.ctypes.data, packed.ctypes.data, values.size)
    surrogate = np.empty((values.size // 32, 34), dtype=np.uint8)
    surrogate[:, :2] = packed[:, :2]
    surrogate[:, 2:] = packed[:, 4:]
    reference = np.empty(values.size, dtype=np.float32)
    dequantize(surrogate.ctypes.data, reference.ctypes.data, reference.size)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "q8_1_qbytes.npy", packed.reshape(-1))
    np.save(args.output / "q8_1_deq.npy", reference)
    provenance = {
        "llama_cpp_commit": revision,
        "shape": list(values.shape),
        "seed": seed,
        "quantizer": "quantize_row_q8_1_ref",
        "oracle": "dequantize_row_q8_0 after copying Q8_1 d and qs into Q8_0 blocks",
        "qbytes_sha256": hashlib.sha256(packed.tobytes()).hexdigest(),
        "deq_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
    }
    (args.output / "q8_1_fixture.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
