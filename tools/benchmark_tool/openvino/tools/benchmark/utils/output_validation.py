# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ValidationThresholds:
    max_ulps_fp32: int = 4
    max_ulps_fp16: int = 64
    max_ulps_bf16: int = 128
    max_abs_fp32: float = 1e-4
    max_rel_fp32: float = 1e-3
    max_abs_fp16: float = 1e-2
    max_rel_fp16: float = 5e-2
    max_abs_bf16: float = 2e-2
    max_rel_bf16: float = 1e-1
    min_cosine_fp32: float = 0.999
    min_cosine_fp16: float = 0.995
    min_cosine_bf16: float = 0.99
    max_abs_int: int = 1
    min_cosine_int: float = 0.999


@dataclass
class OutputValidationResult:
    name: str
    dtype_name: str
    passed: bool
    cosine_similarity: float
    max_ulp_diff: Optional[int] = None
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    failure_reason: str = ""
    sample_mismatch_indices: Optional[List[List[int]]] = None


@dataclass
class ValidationSummary:
    passed: bool
    output_results: List[OutputValidationResult]


def capture_request_inputs(infer_request, num_inputs: int) -> Dict[int, np.ndarray]:
    inputs: Dict[int, np.ndarray] = {}
    for input_idx in range(num_inputs):
        input_tensor = infer_request.get_input_tensor(input_idx)
        inputs[input_idx] = np.array(input_tensor.data, copy=True)
    return inputs


def run_single_inference(compiled_model, input_tensors: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
    request = compiled_model.create_infer_request()

    for input_idx, input_data in input_tensors.items():
        input_tensor = request.get_input_tensor(input_idx)
        if list(input_tensor.shape) != list(input_data.shape):
            input_tensor.shape = input_data.shape
        if not len(input_tensor.shape):
            input_tensor.data.flat[:] = input_data
        else:
            input_tensor.data[:] = input_data

    request.infer()

    outputs: List[Dict[str, Any]] = []
    for output_idx, output_port in enumerate(compiled_model.outputs):
        output_tensor = request.get_output_tensor(output_idx)
        output_name = output_port.any_name if output_port.get_names() else output_port.node.get_friendly_name()
        outputs.append({
            "name": output_name,
            "dtype_name": output_port.element_type.get_type_name(),
            "data": np.array(output_tensor.data, copy=True),
        })

    return outputs


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1).astype(np.float64)
    b_flat = b.reshape(-1).astype(np.float64)

    a_norm = np.linalg.norm(a_flat)
    b_norm = np.linalg.norm(b_flat)

    if a_norm == 0.0 and b_norm == 0.0:
        return 1.0
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    return float(np.dot(a_flat, b_flat) / (a_norm * b_norm))


def max_ulp_diff_float32(a: np.ndarray, b: np.ndarray) -> int:
    a_u = a.astype(np.float32).view(np.uint32)
    b_u = b.astype(np.float32).view(np.uint32)

    a_i = a_u.astype(np.int64)
    b_i = b_u.astype(np.int64)

    # Convert IEEE 754 bit patterns to lexicographically ordered integers.
    sign_mask = 1 << 31
    a_i = np.where(a_i & sign_mask, sign_mask - a_i, a_i)
    b_i = np.where(b_i & sign_mask, sign_mask - b_i, b_i)

    return int(np.max(np.abs(a_i - b_i)))


def max_ulp_diff_float16(a: np.ndarray, b: np.ndarray) -> int:
    a_u = a.astype(np.float16).view(np.uint16)
    b_u = b.astype(np.float16).view(np.uint16)

    a_i = a_u.astype(np.int32)
    b_i = b_u.astype(np.int32)

    sign_mask = 1 << 15
    a_i = np.where(a_i & sign_mask, sign_mask - a_i, a_i)
    b_i = np.where(b_i & sign_mask, sign_mask - b_i, b_i)

    return int(np.max(np.abs(a_i - b_i)))


def get_mismatch_indices(a: np.ndarray, b: np.ndarray, limit: int = 5) -> List[List[int]]:
    diff = np.not_equal(a, b)
    mismatch_idx = np.argwhere(diff)
    return [idx.tolist() for idx in mismatch_idx[:limit]]


def calculate_abs_rel_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    abs_diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0

    denom = np.maximum(np.maximum(np.abs(a.astype(np.float64)), np.abs(b.astype(np.float64))), np.finfo(np.float32).eps)
    rel_diff = abs_diff / denom
    max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0

    return max_abs, max_rel


def validate_outputs_against_reference(target_outputs: List[Dict[str, Any]],
                                       reference_outputs: List[Dict[str, Any]],
                                       thresholds: ValidationThresholds) -> ValidationSummary:
    if len(target_outputs) != len(reference_outputs):
        raise RuntimeError(
            f"Validation failed: output count mismatch: target={len(target_outputs)} reference={len(reference_outputs)}"
        )

    results: List[OutputValidationResult] = []

    for idx, (target_out, reference_out) in enumerate(zip(target_outputs, reference_outputs)):
        target = target_out["data"]
        reference = reference_out["data"]
        name = target_out["name"] or reference_out["name"] or f"output_{idx}"
        dtype_name = target_out["dtype_name"]

        if target.shape != reference.shape:
            results.append(OutputValidationResult(
                name=name,
                dtype_name=dtype_name,
                passed=False,
                cosine_similarity=0.0,
                failure_reason=f"shape mismatch: target={target.shape} reference={reference.shape}",
            ))
            continue

        target_finite = np.nan_to_num(target.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        reference_finite = np.nan_to_num(reference.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        cos = cosine_similarity(target_finite, reference_finite)

        is_float_output = np.issubdtype(target.dtype, np.floating)

        if is_float_output:
            if dtype_name == "f16":
                ulp = max_ulp_diff_float16(target, reference)
                max_abs, max_rel = calculate_abs_rel_diff(target, reference)
                passed = cos >= thresholds.min_cosine_fp16 and (
                    max_abs <= thresholds.max_abs_fp16 or max_rel <= thresholds.max_rel_fp16
                )
                reason = "" if passed else (
                    f"cosine={cos:.8f} (limit={thresholds.min_cosine_fp16}), "
                    f"max_abs_diff={max_abs} (limit={thresholds.max_abs_fp16}), "
                    f"max_rel_diff={max_rel} (limit={thresholds.max_rel_fp16}), "
                    f"ULP={ulp} (diagnostic limit={thresholds.max_ulps_fp16})"
                )
            elif dtype_name == "bf16":
                # NumPy has no native bf16 scalar type in stable APIs.
                # Use float32 ULP on converted values as a practical proxy.
                ulp = max_ulp_diff_float32(target.astype(np.float32), reference.astype(np.float32))
                max_abs, max_rel = calculate_abs_rel_diff(target, reference)
                passed = cos >= thresholds.min_cosine_bf16 and (
                    max_abs <= thresholds.max_abs_bf16 or max_rel <= thresholds.max_rel_bf16
                )
                reason = "" if passed else (
                    f"cosine={cos:.8f} (limit={thresholds.min_cosine_bf16}), "
                    f"max_abs_diff={max_abs} (limit={thresholds.max_abs_bf16}), "
                    f"max_rel_diff={max_rel} (limit={thresholds.max_rel_bf16}), "
                    f"ULP={ulp} (diagnostic limit={thresholds.max_ulps_bf16})"
                )
            else:
                ulp = max_ulp_diff_float32(target.astype(np.float32), reference.astype(np.float32))
                max_abs, max_rel = calculate_abs_rel_diff(target, reference)
                passed = cos >= thresholds.min_cosine_fp32 and (
                    max_abs <= thresholds.max_abs_fp32 or max_rel <= thresholds.max_rel_fp32
                )
                reason = "" if passed else (
                    f"cosine={cos:.8f} (limit={thresholds.min_cosine_fp32}), "
                    f"max_abs_diff={max_abs} (limit={thresholds.max_abs_fp32}), "
                    f"max_rel_diff={max_rel} (limit={thresholds.max_rel_fp32}), "
                    f"ULP={ulp} (diagnostic limit={thresholds.max_ulps_fp32})"
                )

            results.append(OutputValidationResult(
                name=name,
                dtype_name=dtype_name,
                passed=passed,
                cosine_similarity=cos,
                max_ulp_diff=ulp,
                max_abs_diff=max_abs,
                max_rel_diff=max_rel,
                failure_reason=reason,
                sample_mismatch_indices=get_mismatch_indices(target, reference) if not passed else None,
            ))
            continue

        target_i = target.astype(np.int64)
        reference_i = reference.astype(np.int64)
        max_abs_diff = float(np.max(np.abs(target_i - reference_i))) if target_i.size else 0.0
        passed = max_abs_diff <= thresholds.max_abs_int and cos >= thresholds.min_cosine_int
        reason = "" if passed else (
            f"max_abs_diff={max_abs_diff} (limit={thresholds.max_abs_int}), cosine={cos:.8f} "
            f"(limit={thresholds.min_cosine_int})"
        )

        results.append(OutputValidationResult(
            name=name,
            dtype_name=dtype_name,
            passed=passed,
            cosine_similarity=cos,
            max_abs_diff=max_abs_diff,
            failure_reason=reason,
            sample_mismatch_indices=get_mismatch_indices(target, reference) if not passed else None,
        ))

    return ValidationSummary(
        passed=all(result.passed for result in results),
        output_results=results,
    )
