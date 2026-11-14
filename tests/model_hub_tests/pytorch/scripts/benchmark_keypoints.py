# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_keypoint_detectors import MODEL_INPUTS, TestKeypointDetectorsConvertModel
from torch_utils import flattenize_structure


def measure(call, seconds):
    samples = []
    deadline = time.perf_counter() + seconds
    while len(samples) < 5 or time.perf_counter() < deadline:
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def file_fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths = {line.split()[-1] for line in maps.read_text().splitlines()
             if ".so" in line and any(name in line for name in ("openvino", "libtorch", "libigdrcl", "libze_"))}
    return [file_fingerprint(path) for path in sorted(paths) if Path(path).is_file()]


def image_tensor(path, shape):
    _, channels, height, width = shape
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    if channels == 1:
        tensor = tensor.mean(1, keepdim=True)
    return tensor.contiguous()


def benchmark(name, shape, args):
    case = TestKeypointDetectorsConvertModel()
    case.mode = args.mode
    model = case.load_model(name, shape)
    examples = [image_tensor(path, shape) for path in args.images]
    case.example = (examples[0],)
    start = time.perf_counter()
    ov_model = case.convert_model(model)
    conversion_seconds = time.perf_counter() - start
    config = {
        "INFERENCE_PRECISION_HINT": args.ov_precision.lower().replace("fp", "f"),
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": ov.properties.streams.Num(1),
    }
    if args.device == "CPU":
        config.update(INFERENCE_NUM_THREADS=args.threads, ENABLE_CPU_PINNING=False)
    core = ov.Core()
    start = time.perf_counter()
    compiled = core.compile_model(ov_model, args.device, config)
    compilation_seconds = time.perf_counter() - start
    request = compiled.create_infer_request()
    numpy_inputs = [image.numpy() for image in examples]
    output_shapes = []
    accuracy_errors = []
    for image, data in zip(examples, numpy_inputs):
        reference = flattenize_structure(model(image))
        values = list(reference.values()) if isinstance(reference, dict) else reference
        result = request.infer([data], share_inputs=True, share_outputs=True)
        assert len(values) == len(compiled.outputs)
        keys = reference if isinstance(reference, dict) else range(len(reference))
        actual = {key: result[port] for key, port in zip(keys, compiled.outputs)}
        if name == "XFeat":
            # Nearly equal scores can swap ranks while preserving the same feature set.
            for outputs in (reference, actual):
                points = outputs["keypoints"]
                order = np.lexsort((points[:, 1], points[:, 0]))
                for key in outputs:
                    outputs[key] = outputs[key][order]
        entries = reference.items() if isinstance(reference, dict) else enumerate(reference)
        try:
            for key, expected in entries:
                assert actual[key].shape == expected.shape, f"Output {key} shapes differ"
                assert np.isfinite(expected).all() and np.isfinite(actual[key]).all()
                np.testing.assert_allclose(actual[key], expected, atol=5e-2, rtol=5e-2)
            assert values[0].size > 0, "Benchmark image produced no keypoints"
        except AssertionError as error:
            if not args.allow_accuracy_failures:
                raise
            accuracy_errors.append({"image": str(args.images[len(output_shapes)]), "error": str(error)})
        output_shapes.append([list(value.shape) for value in values])

    index = 0

    def infer_torch():
        nonlocal index
        result = model(examples[index % len(examples)])
        index += 1
        return result

    def infer_openvino():
        nonlocal index
        result = request.infer([numpy_inputs[index % len(examples)]], share_inputs=True, share_outputs=True)
        index += 1
        return result

    for _ in range(args.warmup):
        infer_torch()
        infer_openvino()
    timings = {"torch": [], "openvino": []}
    rounds = []
    for round_index in range(args.rounds):
        order = [("torch", infer_torch), ("openvino", infer_openvino)]
        if round_index % 2:
            order.reverse()
        summary = {}
        for engine, call in order:
            index = 0
            samples = measure(call, args.seconds)
            timings[engine].extend(samples)
            summary[engine] = statistics.median(samples)
        rounds.append(summary)
    result = {
        "model": name,
        "shape": list(shape),
        "mode": args.mode,
        "device": args.device,
        "device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
        "openvino_precision": args.ov_precision,
        "execution_devices": compiled.get_property("EXECUTION_DEVICES"),
        "inference_precision_hint": str(compiled.get_property("INFERENCE_PRECISION_HINT")),
        "compilation_config": {key: 1 if key == "NUM_STREAMS" else value for key, value in config.items()},
        "conversion_seconds": conversion_seconds,
        "compilation_seconds": compilation_seconds,
        "output_shapes": output_shapes,
        "accuracy_passed": not accuracy_errors,
        "accuracy_errors": accuracy_errors,
        "accuracy_order": "coordinates" if name == "XFeat" else "model outputs",
        "round_medians_ms": rounds,
    }
    for engine, samples in timings.items():
        result[engine] = {
            "median_ms": statistics.median(round_[engine] for round_ in rounds),
            "p90_ms": float(np.percentile(samples, 90)),
            "iterations": len(samples),
            "samples_ms": samples,
        }
    result["speedup"] = result["torch"]["median_ms"] / result["openvino"]["median_ms"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare full keypoint inference with PyTorch CPU FP32.")
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=sorted({name for name, _ in MODEL_INPUTS}))
    parser.add_argument("--shapes", nargs="+", help="Override tested shapes, for example 256x320 480x640")
    parser.add_argument("--mode", choices=["trace", "export"], default="export")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--ov-precision", choices=["FP32", "FP16"], default="FP32")
    parser.add_argument("--allow-accuracy-failures", action="store_true",
                        help="Retain failed accuracy checks and measure latency for investigation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=0.2, help="Minimum measurement time per engine per round")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.seconds) <= 0 or args.warmup < 0:
        parser.error("Threads, rounds and seconds must be positive; warmup must be nonnegative")
    unknown_models = set(args.models) - {name for name, _ in MODEL_INPUTS}
    if unknown_models:
        parser.error(f"Unknown models: {sorted(unknown_models)}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    source = Path(__file__).resolve().parents[1] / "test_keypoint_detectors.py"
    report = {
        "metadata": {
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch": torch.__version__, "openvino": ov.__version__,
            "torch_module": torch.__file__, "openvino_module": ov.__file__,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "baseline": "PyTorch eager, inference_mode, contiguous NCHW",
            "torch_precision": "FP32", "precision": args.ov_precision, "device": args.device, "streams": 1,
            "warmup": args.warmup, "rounds": args.rounds, "seconds_per_round": args.seconds,
            "sources": [file_fingerprint(source), file_fingerprint(__file__)],
            "images": [file_fingerprint(path) for path in args.images],
            "accuracy_atol": 5e-2, "accuracy_rtol": 5e-2,
            "allow_accuracy_failures": args.allow_accuracy_failures,
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [(name, shape) for name, shape in MODEL_INPUTS if name in args.models]
    if args.shapes:
        shapes = [(1, 3, *map(int, shape.split("x"))) for shape in args.shapes]
        cases = [(name, shape) for name in args.models for shape in shapes]
    failures = 0
    with torch.inference_mode():
        for name, shape in cases:
            try:
                result = benchmark(name, shape, args)
                print(f"{name} {shape}: torch={result['torch']['median_ms']:.3f} ms "
                      f"openvino={result['openvino']['median_ms']:.3f} ms speedup={result['speedup']:.2f}x "
                      f"accuracy={result['accuracy_passed']}", flush=True)
                failures += not result["accuracy_passed"]
            except Exception as error:
                traceback.print_exc()
                failures += 1
                result = {"model": name, "shape": list(shape), "mode": args.mode, "error": str(error)}
            report["results"].append(result)
            if "loaded_libraries" not in report["metadata"]:
                report["metadata"]["loaded_libraries"] = loaded_libraries()
            args.output.write_text(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
