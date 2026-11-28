# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Pack sparse-selection and hyper-connection references from real ggml CPU into an NPZ.

Build sparse_ops_oracle.cpp against a CPU-only llama.cpp checkout, then run from this directory:
  python gen_sparse_ops_accuracy.py --oracle <sparse_ops_oracle>
OpenVINO does not participate in reference generation.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

TYPES = {"f32": np.float32, "f16": np.float16, "i32": np.int32}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "test_data/sparse_ops.npz")
    args = parser.parse_args()
    arrays = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([args.oracle, tmp], check=True)
        for shape_file in sorted(Path(tmp).glob("*.shape")):
            kind, *dims = shape_file.read_text().split()
            data = np.fromfile(shape_file.with_suffix(".bin"), TYPES[kind])
            arrays[shape_file.stem] = data.reshape([int(d) for d in dims])
    np.savez_compressed(args.output, **arrays)
    print(f"{args.output}: {len(arrays)} arrays")


if __name__ == "__main__":
    main()
