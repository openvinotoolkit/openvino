# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare a real encoder checkpoint against the pinned llama.cpp CPU encoder.

Build mmproj_oracle.cpp against llama.cpp 16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb.
Inputs are normalized encoder tensors; media preprocessing is validated separately.
For quantized checkpoints, pass --reference-model with an F32 copy produced by
dequantize_mmproj.py. Omitting it compares the same file on both runtimes, which
also measures differences in their quantized execution kernels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager

from mmproj_fixtures import checkpoint_inputs, run_oracle


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oracle-revision", default="16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb",
                        help="Exact llama.cpp revision used to build the supplied oracle")
    parser.add_argument("--reference-model", type=Path,
                        help="llama.cpp reference checkpoint; use an F32 copy of the represented weights")
    parser.add_argument("--nmse-threshold", type=float, default=1e-5,
                        help="Strict upper bound on normalized MSE (default: 1e-5)")
    parser.add_argument("--modality", choices=["vision", "audio"], default="vision")
    parser.add_argument("--width", type=int, help="Image width or audio feature-frame count")
    parser.add_argument("--height", type=int, help="Image height")
    args = parser.parse_args()
    if not np.isfinite(args.nmse_threshold) or args.nmse_threshold <= 0:
        parser.error("--nmse-threshold must be finite and positive")
    reference_model = args.reference_model or args.model
    frontend = FrontEndManager().load_by_framework("gguf")
    model = frontend.convert(frontend.load(str(args.model)))
    metadata = model.get_rt_info(["gguf_mmproj"]).value
    projector = metadata[args.modality + ".projector"]
    # Extract only the reachable modality, matching AdaptMmprojToGenAI branch selection.
    output = model.output(args.modality + ".embeddings")
    reachable = set()
    def visit(node):
        if node in reachable:
            return
        reachable.add(node)
        for value in node.input_values():
            visit(value.get_node())
    visit(output.get_node())
    model = ov.Model([output], [p for p in model.get_parameters() if p in reachable])
    if args.modality == "vision":
        default = int(metadata["clip.vision.image_size"])
        width, height = args.width or default, args.height or default
    else:
        width, height = args.width or (9 if projector == "gemma4ua" else 101), None
    feeds, raw, width, height = checkpoint_inputs(metadata, args.modality, width, height)
    shape = list(next(v for k, v in feeds.items() if k.endswith(("pixel_values", "features", "waveform_frames"))).shape)
    expected = run_oracle(args.oracle, reference_model.resolve(), args.modality, width, height, raw)
    request = ov.Core().compile_model(model, "CPU", {
        "INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
        "INFERENCE_NUM_THREADS": 4,
    }).create_infer_request()
    request.infer(feeds)
    actual = request.get_output_tensor().data.reshape(-1).copy()
    assert actual.shape == expected.shape
    error = actual.astype(np.float64) - expected
    squared_error = float(np.dot(error, error))
    reference_energy = float(np.dot(expected.astype(np.float64), expected))
    nmse = squared_error / reference_energy if reference_energy > 0 else (0.0 if squared_error == 0 else float("inf"))
    model_digest = sha256(args.model)
    reference_digest = model_digest if reference_model.resolve() == args.model.resolve() else sha256(reference_model)
    report = {"model": str(args.model.resolve()), "sha256": model_digest,
              "reference_model": str(reference_model.resolve()), "reference_sha256": reference_digest,
              "reference_revision": args.oracle_revision,
              "openvino_version": ov.get_version(), "input_shape": shape,
              "q4k_f16_zero_point": os.getenv("OV_GGUF_Q4_K_ZP_F16", "0") not in {"", "0"},
              "modality": args.modality, "projector": projector,
              "normalized_mse": nmse, "max_absolute_error": float(np.max(np.abs(error))),
              "nmse_threshold": args.nmse_threshold,
              "passed": bool(np.isfinite(nmse) and nmse < args.nmse_threshold)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
