# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare a real vision checkpoint against the pinned llama.cpp CPU encoder.

Build mmproj_oracle.cpp against llama.cpp 16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb.
Inputs are normalized encoder tensors; media preprocessing is validated separately.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frontend = FrontEndManager().load_by_framework("gguf")
    model = frontend.convert(frontend.load(str(args.model)))
    shape = list(model.input("vision.pixel_values").shape)
    rng = np.random.default_rng(42)
    values = rng.uniform(-1, 1, shape).astype(np.float32)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        values[0].transpose(1, 2, 0).tofile(directory / "input.f32")
        subprocess.run([str(args.oracle.resolve()), str(args.model.resolve()), "vision",
                        str(shape[3]), str(shape[2]), str(directory / "input.f32"),
                        str(directory / "output.f32")], check=True)
        expected = np.fromfile(directory / "output.f32", dtype=np.float32)
    request = ov.Core().compile_model(model, "CPU", {
        "INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        "INFERENCE_NUM_THREADS": 4,
    }).create_infer_request()
    request.infer({"vision.pixel_values": values})
    actual = request.get_tensor("vision.embeddings").data.reshape(-1).copy()
    assert actual.shape == expected.shape
    error = actual.astype(np.float64) - expected
    nmse = float(np.dot(error, error) / np.dot(expected.astype(np.float64), expected))
    with args.model.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    report = {"model": str(args.model.resolve()), "sha256": digest.hexdigest(),
              "reference_revision": "16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb",
              "openvino_version": ov.get_version(), "input_shape": shape,
              "normalized_mse": nmse, "max_absolute_error": float(np.max(np.abs(error))),
              "passed": bool(np.isfinite(nmse) and nmse < 1e-5)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
