# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Generate MTP logits/hidden states using CPU-only llama.cpp, revision
5047748f114564fcf271e6a648f60e2d2eae6264.

Build mtp_oracle.cpp with include/, src/, ggml/include/ and link libllama,
libggml, libggml-base from that revision. Run with --oracle <binary>.
The resulting small fixture is consumed without llama.cpp at test time.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from gen_arch_accuracy import write_qwen35_model
from mmproj_fixtures import save_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        model = Path(directory) / "mtp.gguf"
        output = Path(directory) / "reference.bin"
        write_qwen35_model(model, "qwen35", nextn=True)
        subprocess.run([str(args.oracle.resolve()), str(model), str(output)], check=True)
        raw = np.fromfile(output, dtype=np.float32)
        arrays = {"model": np.frombuffer(model.read_bytes(), dtype=np.uint8).copy()}
        cursor = 0
        for step, length in enumerate((3, 1, 2)):
            count = length * 32
            arrays[f"hidden_input_{step}"] = raw[cursor:cursor + count].reshape(1, length, 32)
            cursor += count
            rows = raw[cursor:cursor + 2 * count].reshape(length, 64)
            arrays[f"logits_{step}"] = rows[:, :32].reshape(1, length, 32)
            arrays[f"hidden_output_{step}"] = rows[:, 32:].reshape(1, length, 32)
            cursor += 2 * count
        assert cursor == len(raw), "Unexpected MTP oracle output size"
        destination = Path(__file__).parent / "test_data" / "arch_accuracy" / "qwen35-mtp.npz"
        save_npz(destination, arrays)


if __name__ == "__main__":
    main()
