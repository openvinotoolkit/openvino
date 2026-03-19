#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import openvino as ov


def load_meta(path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def load_indices(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size == 0:
        raise RuntimeError(f"indices file is empty: {path}")
    count = int(raw[0])
    values = raw[1:]
    if values.size != count:
        raise RuntimeError(f"indices count mismatch in {path}: header={count}, payload={values.size}")
    return values


def load_intervals(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise RuntimeError(f"intervals file is empty: {path}")
    count = int(raw[0])
    payload = raw[1:]
    if payload.size != count * 3:
        raise RuntimeError(
            f"intervals count mismatch in {path}: header={count}, payload_triplets={payload.size // 3}, payload_size={payload.size}"
        )
    return payload.reshape(count, 3)


def cast_for_input(x: np.ndarray, element_type: ov.Type) -> np.ndarray:
    if element_type == ov.Type.f16:
        return x.astype(np.float16, copy=False)
    if element_type == ov.Type.f32:
        return x.astype(np.float32, copy=False)
    if element_type == ov.Type.i32:
        return x.astype(np.int32, copy=False)
    if element_type == ov.Type.i64:
        return x.astype(np.int64, copy=False)
    if element_type == ov.Type.u32:
        return x.astype(np.uint32, copy=False)
    raise RuntimeError(f"Unsupported model input element type: {element_type}")


def _is_float_type(element_type: ov.Type) -> bool:
    return element_type in {ov.Type.f16, ov.Type.f32, ov.Type.f64, ov.Type.bf16}


def _is_int_type(element_type: ov.Type) -> bool:
    return element_type in {
        ov.Type.i8,
        ov.Type.i16,
        ov.Type.i32,
        ov.Type.i64,
        ov.Type.u8,
        ov.Type.u16,
        ov.Type.u32,
        ov.Type.u64,
    }


def resolve_model_inputs(model: ov.Model) -> Dict[str, ov.Output]:
    by_name: Dict[str, ov.Output] = {}
    for port in model.inputs:
        try:
            for name in port.get_names():
                by_name[name] = port
        except Exception:
            pass
        try:
            by_name[port.get_any_name()] = port
        except Exception:
            pass

    name_alias = {
        "feat": ["feat", "features", "camera_features"],
        "depth": ["depth", "depth_weights", "camera_depth_weights"],
        "indices": ["indices", "ranks_depth", "ranks"],
        "intervals": ["intervals", "itv", "interval"],
    }

    resolved: Dict[str, ov.Output] = {}
    used_ids = set()
    for logical, aliases in name_alias.items():
        for alias in aliases:
            port = by_name.get(alias)
            if port is not None and id(port) not in used_ids:
                resolved[logical] = port
                used_ids.add(id(port))
                break

    if len(resolved) == 4:
        return resolved

    remaining: List[ov.Output] = [port for port in model.inputs if id(port) not in used_ids]

    float_ports = [p for p in remaining if _is_float_type(p.get_element_type())]
    int_ports = [p for p in remaining if _is_int_type(p.get_element_type())]

    if "feat" not in resolved or "depth" not in resolved:
        if len(float_ports) >= 2:
            rank4_float = [p for p in float_ports if p.get_partial_shape().rank.is_static and p.get_partial_shape().rank.get_length() == 4]
            chosen = rank4_float[:2] if len(rank4_float) >= 2 else float_ports[:2]
            if "feat" not in resolved:
                resolved["feat"] = chosen[0]
                used_ids.add(id(chosen[0]))
            if "depth" not in resolved:
                resolved["depth"] = chosen[1]
                used_ids.add(id(chosen[1]))

    remaining = [port for port in model.inputs if id(port) not in used_ids]
    int_ports = [p for p in remaining if _is_int_type(p.get_element_type())]

    if "indices" not in resolved or "intervals" not in resolved:
        if len(int_ports) >= 2:
            rank1 = [p for p in int_ports if p.get_partial_shape().rank.is_static and p.get_partial_shape().rank.get_length() == 1]
            rank2 = [p for p in int_ports if p.get_partial_shape().rank.is_static and p.get_partial_shape().rank.get_length() == 2]

            if "indices" not in resolved:
                if rank1:
                    resolved["indices"] = rank1[0]
                else:
                    resolved["indices"] = int_ports[0]
                used_ids.add(id(resolved["indices"]))

            if "intervals" not in resolved:
                candidates = [p for p in int_ports if id(p) != id(resolved.get("indices"))]
                if rank2:
                    for p in rank2:
                        if id(p) != id(resolved.get("indices")):
                            resolved["intervals"] = p
                            break
                if "intervals" not in resolved and candidates:
                    resolved["intervals"] = candidates[0]
                if "intervals" in resolved:
                    used_ids.add(id(resolved["intervals"]))

    missing = [k for k in ["feat", "depth", "indices", "intervals"] if k not in resolved]
    if missing:
        input_desc = []
        for i, p in enumerate(model.inputs):
            n = ""
            try:
                n = p.get_any_name()
            except Exception:
                n = "<noname>"
            input_desc.append(f"#{i}:{n}:{p.get_element_type()}:{p.get_partial_shape()}")
        raise RuntimeError(
            "Failed to resolve model inputs "
            + str(missing)
            + ". model inputs: "
            + "; ".join(input_desc)
            + ". This comparator expects a 4-input BevPoolV2 model (feat/depth/indices/intervals)."
        )

    return resolved


def infer_one_device(
    core: ov.Core,
    model_path: Path,
    device: str,
    feat: np.ndarray,
    depth: np.ndarray,
    indices: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    model = core.read_model(str(model_path))
    resolved = resolve_model_inputs(model)

    reshape_map = {
        resolved["feat"]: ov.PartialShape(list(feat.shape)),
        resolved["depth"]: ov.PartialShape(list(depth.shape)),
        resolved["indices"]: ov.PartialShape(list(indices.shape)),
        resolved["intervals"]: ov.PartialShape(list(intervals.shape)),
    }
    model.reshape(reshape_map)

    compiled = core.compile_model(model, device)

    model_to_index = {id(port): i for i, port in enumerate(model.inputs)}
    in_feat = compiled.input(model_to_index[id(resolved["feat"])])
    in_depth = compiled.input(model_to_index[id(resolved["depth"])])
    in_indices = compiled.input(model_to_index[id(resolved["indices"])])
    in_intervals = compiled.input(model_to_index[id(resolved["intervals"])])

    request_inputs = {
        in_feat: cast_for_input(feat, in_feat.get_element_type()),
        in_depth: cast_for_input(depth, in_depth.get_element_type()),
        in_indices: cast_for_input(indices, in_indices.get_element_type()),
        in_intervals: cast_for_input(intervals, in_intervals.get_element_type()),
    }

    req = compiled.create_infer_request()
    req.infer(request_inputs)
    out = req.get_output_tensor(0).data
    return np.array(out, copy=True)


def error_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    abs_diff = np.abs(diff)
    denom = np.maximum(np.abs(b.astype(np.float64)), 1e-12)
    rel_diff = abs_diff / denom
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
    }


def print_stats(tag: str, stats: Dict[str, float]) -> None:
    print(
        f"[{tag}] max_abs={stats['max_abs']:.6e}, "
        f"mean_abs={stats['mean_abs']:.6e}, "
        f"max_rel={stats['max_rel']:.6e}, "
        f"mean_rel={stats['mean_rel']:.6e}"
    )


def topk_mismatch(a: np.ndarray, b: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
    k = max(1, min(k, diff.size))
    idx = np.argpartition(diff, -k)[-k:]
    idx = idx[np.argsort(diff[idx])[::-1]]
    return idx, diff[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SYCL bevPool reference bins with OpenVINO CPU/GPU outputs")
    parser.add_argument("--model", required=True, help="Path to ONNX/IR model")
    parser.add_argument("--ref-dir", required=True, help="Directory containing generated bin files")
    parser.add_argument("--topk", type=int, default=10, help="Print top-k worst absolute mismatches")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    ref_dir = Path(args.ref_dir).resolve()

    meta = load_meta(ref_dir / "meta.txt")
    n = int(meta["n"])
    c = int(meta["c"])
    d = int(meta["d"])
    h = int(meta["h"])
    w = int(meta["w"])
    bev_h = int(meta["bev_h"])
    bev_w = int(meta["bev_w"])

    feat = np.fromfile(ref_dir / "camera_features.bin", dtype=np.float32).reshape(n, h, w, c)
    depth = np.fromfile(ref_dir / "camera_depth_weights.bin", dtype=np.float32).reshape(n, d, h, w)
    indices = load_indices(ref_dir / "indices.bin")
    intervals = load_intervals(ref_dir / "intervals.bin")
    ref_out = np.fromfile(ref_dir / "bev_ref_output.bin", dtype=np.float32).reshape(n, c, bev_h, bev_w)

    print(f"Model: {model_path}")
    print(f"Ref dir: {ref_dir}")
    print(f"feat={feat.shape}, depth={depth.shape}, indices={indices.shape}, intervals={intervals.shape}, ref_out={ref_out.shape}")

    core = ov.Core()
    cpu_out = infer_one_device(core, model_path, "CPU", feat, depth, indices, intervals)
    print(f"CPU output shape: {cpu_out.shape}")
    cpu_stats = error_stats(cpu_out, ref_out)
    print_stats("CPU vs REF", cpu_stats)

    gpu_out = None
    available_devices = set(core.available_devices)
    if "GPU" in available_devices:
        gpu_out = infer_one_device(core, model_path, "GPU", feat, depth, indices, intervals)
        print(f"GPU output shape: {gpu_out.shape}")
        gpu_stats = error_stats(gpu_out, ref_out)
        print_stats("GPU vs REF", gpu_stats)

        cpu_gpu_stats = error_stats(cpu_out, gpu_out)
        print_stats("CPU vs GPU", cpu_gpu_stats)
    else:
        print("GPU device is not available in OpenVINO runtime; skipped GPU comparison")

    idx, vals = topk_mismatch(cpu_out, ref_out, args.topk)
    print(f"Top-{len(idx)} CPU vs REF abs mismatches (flatten index, abs_diff):")
    for i, v in zip(idx.tolist(), vals.tolist()):
        print(f"  {i}: {v:.6e}")

    if gpu_out is not None:
        idx_g, vals_g = topk_mismatch(gpu_out, ref_out, args.topk)
        print(f"Top-{len(idx_g)} GPU vs REF abs mismatches (flatten index, abs_diff):")
        for i, v in zip(idx_g.tolist(), vals_g.tolist()):
            print(f"  {i}: {v:.6e}")


if __name__ == "__main__":
    main()
