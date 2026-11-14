# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Generate MTP draft and target logits/hidden states using CPU-only llama.cpp: revision
5047748f114564fcf271e6a648f60e2d2eae6264 for qwen35 and
0ec4027a2a0183de87c007069adddc06e665f7de for qwen4exp.

Build mtp_oracle.cpp with include/, src/, ggml/include/ and link libllama,
libggml, libggml-base from that revision. Run with --oracle <binary> and
--architectures. The resulting small fixtures are consumed without llama.cpp
at test time.
"""

import argparse
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from gen_arch_accuracy import write_qwen35_model, write_qwen4exp_model
from mmproj_fixtures import save_npz


WRITERS = {
    "qwen35": (lambda path: write_qwen35_model(path, "qwen35", nextn=True), 32, 32),
    "qwen4exp": (lambda path: write_qwen4exp_model(path, nextn=True), 32, 64),
}
LENGTHS = (3, 1, 2)


def run(oracle, model, output, vocab, width):
    subprocess.run([str(oracle), str(model), str(output)], check=True)
    raw = np.fromfile(output, dtype=np.float32)
    arrays, cursor = {}, 0
    for step, length in enumerate(LENGTHS):
        arrays[f"hidden_input_{step}"] = raw[cursor:cursor + length * width].reshape(1, length, width)
        cursor += length * width
        rows = raw[cursor:cursor + length * (vocab + width)].reshape(length, vocab + width)
        arrays[f"logits_{step}"] = rows[:, :vocab].reshape(1, length, vocab)
        arrays[f"hidden_output_{step}"] = rows[:, vocab:].reshape(1, length, width)
        cursor += length * (vocab + width)
    for step, length in enumerate(LENGTHS):
        arrays[f"target_logits_{step}"] = raw[cursor:cursor + vocab].reshape(1, 1, vocab)
        cursor += vocab
        arrays[f"target_hidden_{step}"] = raw[cursor:cursor + length * width].reshape(1, length, width)
        cursor += length * width
    assert cursor == len(raw), "Unexpected MTP oracle output size"
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--architectures", nargs="+", choices=sorted(WRITERS), default=["qwen35"])
    args = parser.parse_args()
    for arch in args.architectures:
        write, vocab, width = WRITERS[arch]
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "mtp.gguf"
            output = Path(directory) / "reference.bin"
            write(model)
            arrays = {"model": np.frombuffer(model.read_bytes(), dtype=np.uint8).copy()}
            arrays.update(run(args.oracle.resolve(), model, output, vocab, width))
            if arch == "qwen4exp":
                for head in ("self", "shared"):
                    write_qwen4exp_model(model, nextn=True, mtp_head=head)
                    arrays[f"{head}_head_model"] = np.frombuffer(model.read_bytes(), dtype=np.uint8).copy()
            destination = Path(__file__).parent / "test_data" / "arch_accuracy" / f"{arch}-mtp.npz"
            save_npz(destination, arrays)


if __name__ == "__main__":
    main()
