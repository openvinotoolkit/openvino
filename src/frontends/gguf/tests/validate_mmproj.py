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
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
from openvino.frontend import FrontEndManager


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
    rng = np.random.default_rng(42)
    feeds = {}
    if args.modality == "vision":
        if projector not in {"gemma3", "gemma4v", "gemma4uv", "pixtral", "phi4", "qwen3vl_merger"}:
            raise ValueError(f"Add processor/index inputs for {projector} before checkpoint validation")
        default = int(metadata["clip.vision.image_size"])
        width, height = args.width or default, args.height or default
        shape = [1, 3, height, width]
        values = rng.uniform(0 if projector == "gemma4v" else -1, 1, shape).astype(np.float32)
        feeds["vision.pixel_values"] = values
        patch = int(metadata["clip.vision.patch_size"])
        if projector == "gemma4uv":
            patch *= int(metadata.get("clip.vision.projector.scale_factor", 3))
        if projector in {"gemma4v", "gemma4uv", "pixtral"}:
            rows, cols = np.indices((height // patch, width // patch))
            feeds["vision.position_x"] = cols.astype(np.int32).reshape(1, 1, 1, -1)
            feeds["vision.position_y"] = rows.astype(np.int32).reshape(1, 1, 1, -1)
        if projector == "qwen3vl_merger":
            if width % (2 * patch) or height % (2 * patch):
                raise ValueError("Qwen image dimensions must be divisible by twice the patch size")
            gh, gw = height // patch, width // patch
            indices = [y * gw + x + dy * gw + dx
                       for y in range(0, gh, 2) for x in range(0, gw, 2)
                       for dy in range(2) for dx in range(2)]
            rows, cols = np.divmod(indices, gw)
            feeds["vision.pixel_values"] = np.repeat(values, 2, axis=0)
            feeds["vision.patch_indices"] = np.array(indices, np.int32).reshape(1, 1, 1, -1)
            feeds["vision.position_ids"] = np.array([rows, cols, rows, cols], np.int32).reshape(1, 1, 1, -1)
            shape = list(feeds["vision.pixel_values"].shape)
        raw = values[0].transpose(1, 2, 0)
    else:
        if projector not in {"gemma4a", "gemma4ua"}:
            raise ValueError(f"Add audio input contract for {projector}")
        if projector == "gemma4ua":
            width, height = args.width or 9, 640
            raw = rng.normal(0, .4, (height, width)).astype(np.float32)
            shape = [1, 1, width, height]
            feeds["audio.waveform_frames"] = raw.T.reshape(shape)
        else:
            width, height = args.width or 101, int(metadata["clip.audio.num_mel_bins"])
            shape = [1, 1, height, width]
            values = rng.normal(0, .4, shape).astype(np.float32)
            feeds["audio.features"] = values
            channels = int(metadata["clip.audio.embedding_length"])
            n = (width + 3) // 4
            q, k = np.indices((n, n))
            distance = q - k
            timescale = np.exp(-np.arange(channels // 2, dtype=np.float32) *
                               (np.log(np.float32(10000)) / max(channels // 2 - 1, 1)))
            theta = np.arange(12, -1, -1, dtype=np.float32)[:, None] * timescale[None]
            feeds["audio.position_embeddings"] = np.concatenate([np.sin(theta), np.cos(theta)], axis=1)[None, None]
            feeds["audio.attention_mask"] = np.where((distance >= 0) & (distance < 12), 0, -1e9).astype(np.float32)[None, None]
            feeds["audio.relative_indices"] = np.clip(12 - distance, 0, 12).astype(np.int32)[None, None]
            raw = values
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw.tofile(directory / "input.f32")
        subprocess.run([str(args.oracle.resolve()), str(reference_model.resolve()), args.modality,
                        str(width), str(height), str(directory / "input.f32"),
                        str(directory / "output.f32")], check=True)
        expected = np.fromfile(directory / "output.f32", dtype=np.float32)
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
              "reference_revision": "16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb",
              "openvino_version": ov.get_version(), "input_shape": shape,
              "modality": args.modality, "projector": projector,
              "normalized_mse": nmse, "max_absolute_error": float(np.max(np.abs(error))),
              "nmse_threshold": args.nmse_threshold,
              "passed": bool(np.isfinite(nmse) and nmse < args.nmse_threshold)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
