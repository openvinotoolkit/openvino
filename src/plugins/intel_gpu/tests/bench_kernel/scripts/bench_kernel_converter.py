#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
bench_kernel_converter - Convert OV GPU verbose logs to bench_kernel commands.

This tool can:
1. Parse OV GPU bench verbose logs (OV_GPU_BenchVerbose=1) and generate
   bench_kernel command lines for reproducing individual kernel benchmarks.

2. Generate batch files from model descriptions.

Usage:
    # Parse verbose log from benchmark_app
    OV_GPU_BenchVerbose=1 benchmark_app -m model.xml -d GPU -niter 1 2>&1 | \\
        grep ov_gpu_bench > verbose.log
    python3 bench_kernel_converter.py --input verbose.log --output batch_conv.txt --kernel conv

    # Convert all supported kernels
    python3 bench_kernel_converter.py --input verbose.log --output batch_all.txt

    # Generate batch file for common LLM shapes
    python3 bench_kernel_converter.py --generate llm_fc --output fc_batch.txt

    # Parse from stdin (pipe benchmark_app output)
    OV_GPU_BenchVerbose=1 benchmark_app -m model.xml -d GPU -niter 1 2>&1 | \\
        python3 bench_kernel_converter.py --input - --kernel conv --output batch.txt
"""

import argparse
import re
import sys
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict


# ============================================================================
# OV GPU Bench Verbose Log Parser
# ============================================================================

# All known primitive-specific attribute keys from verbose log
PRIMITIVE_ATTR_KEYS = {
    # gemm
    "transpose_a", "transpose_b", "order0", "order1",
    # sdpa
    "is_causal", "order_q", "order_k", "order_v", "order_out", "scale_val",
    # fully_connected
    "compressed", "dynamic_quantized",
    # convolution
    "groups", "strides", "dilations", "padding_begin", "padding_end", "grouped_weights_shape",
    # pooling
    "pool_mode", "kernel", "pool_strides", "pads_begin", "pads_end", "rounding_type",
    # reduce
    "reduce_mode", "keep_dims", "reduce_axes",
    # softmax
    "axis",
    # mvn
    "normalize_variance", "epsilon", "eps_inside_sqrt",
    # eltwise
    "eltwise_mode", "pythondiv",
    # swiglu
    "glu_type", "split_axis", "split_length", "gate_idx",
    # gather
    "gather_axis", "batch_dim", "support_neg_ind",
    # rope
    "head_cnt", "head_size", "rotary_ndims", "is_interleaved", "is_chatglm",
    "is_qwen", "input_trans0213", "slice_start", "slice_stop", "gather_rank",
    # crop
    "offsets",
    # strided_slice
    "ss_begin", "ss_end", "ss_strides", "begin_mask", "end_mask",
    "shrink_axis_mask", "new_axis_mask",
    # concatenation
    "concat_axis",
    # tile
    "tile_repeats",
    # normalize
    "across_spatial",
    # group_normalization
    "num_groups",
    # quantize
    "levels",
    # scatter_nd_update
    "indices_rank",
    # detection_output
    "det_num_classes", "det_keep_top_k", "det_top_k",
    "det_nms_threshold", "det_confidence_threshold",
    "det_code_type", "det_share_location",
    "det_background_label_id", "det_variance_encoded",
    # resample
    "resample_sizes", "resample_mode",
    # permute
    "permute_order",
    # broadcast
    "broadcast_axes", "broadcast_target",
}

class BenchVerboseEntry:
    """Represents a parsed ov_gpu_bench verbose log entry."""
    def __init__(self):
        self.device: str = ""         # e.g. gpu.0, gpu.1
        self.prim_type: str = ""       # e.g. fully_connected, convolution, gemm
        self.prim_id: str = ""         # e.g. fc_0, convolution:Multiply_19322
        self.impl: str = ""            # e.g. ocl, onednn
        self.kernel: str = ""          # e.g. jit:ir, fc_bf_tiled
        self.inputs: List[Tuple[str, List[int]]] = []   # [(dt, [dims]), ...]
        self.outputs: List[Tuple[str, List[int]]] = []   # [(dt, [dims]), ...]
        self.fused_ops: List[str] = []  # e.g. ["activation_relu", "eltwise_sum"]
        self.truncate: bool = False        # reorder truncation mode (Convert op)
        self.time_us: float = 0.0
        # Primitive-specific attributes (parsed from verbose log)
        self.attrs: Dict[str, str] = {}   # key=value pairs from verbose


def split_on_plus(s: str) -> List[str]:
    """Split string on '+' delimiters, respecting scientific notation.

    E.g. 'activation_relu+activation_clamp:-6.78069e+07:6.78069e+07+quantize'
    → ['activation_relu', 'activation_clamp:-6.78069e+07:6.78069e+07', 'quantize']
    """
    parts = []
    current = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '+':
            # Check if this '+' is part of scientific notation (digit(s) + 'e'/'E' + '+')
            if len(current) >= 2:
                prev = current[-1]
                prev2 = current[-2]
                if prev in ('e', 'E') and prev2.isdigit():
                    current.append(ch)
                    i += 1
                    continue
            # It's a genuine delimiter
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append(''.join(current))
    return parts


def parse_dt_shape(dt_shape_str: str) -> Tuple[str, List[int], str]:
    """Parse 'f16:1x64x56x56:bfyx' into ('f16', [1, 64, 56, 56], 'bfyx').

    The format field is optional for backward compatibility:
    'f16:1x64x56x56' -> ('f16', [1, 64, 56, 56], '')
    """
    parts = dt_shape_str.split(":", 2)
    dt = parts[0]
    dims = []
    fmt = ""
    if len(parts) > 1 and parts[1]:
        for d in parts[1].split("x"):
            try:
                dims.append(int(d))
            except ValueError:
                dims.append(-1)  # dynamic dim
    if len(parts) > 2 and parts[2]:
        fmt = parts[2]
    return (dt, dims, fmt)


def parse_bench_verbose_line(line: str) -> Optional[BenchVerboseEntry]:
    """Parse a single ov_gpu_bench verbose log line.

    Format: ov_gpu_bench,exec,type=TYPE,id=ID,impl=IMPL,kernel=KERNEL,
            in0=DT:DxDx...,in1=DT:DxDx...,out0=DT:DxDx...,
            fused=OP+OP,time=TIME_US
    """
    line = line.strip()
    if not line.startswith("ov_gpu_bench,exec,"):
        return None

    entry = BenchVerboseEntry()

    # Parse key=value fields
    # Split carefully - the line starts with "ov_gpu_bench,exec," followed by key=value pairs
    remainder = line[len("ov_gpu_bench,exec,"):]
    fields = remainder.split(",")

    for field in fields:
        if "=" not in field:
            continue
        key, val = field.split("=", 1)
        key = key.strip()
        val = val.strip()

        if key == "device":
            entry.device = val  # e.g. "gpu.0"
        elif key == "type":
            entry.prim_type = val
        elif key == "id":
            entry.prim_id = val
        elif key == "impl":
            entry.impl = val
        elif key == "kernel":
            # "kernel" appears twice for pooling/conv: once for impl kernel name
            # (e.g. "jit:ir", "ocl:ref") and once for pooling kernel size (e.g. "2x2").
            # Detect which one this is and store accordingly.
            if entry.kernel and key in PRIMITIVE_ATTR_KEYS:
                # Already have impl kernel; this is a primitive attribute
                entry.attrs[key] = val
            else:
                entry.kernel = val
        elif key.startswith("in"):
            entry.inputs.append(parse_dt_shape(val))
        elif key.startswith("out"):
            entry.outputs.append(parse_dt_shape(val))
        elif key == "fused":
            if val != "none":
                entry.fused_ops = split_on_plus(val)
        elif key == "truncate":
            entry.truncate = (val == "1")
        elif key == "time":
            try:
                entry.time_us = float(val)
            except ValueError:
                pass
        elif key in PRIMITIVE_ATTR_KEYS:
            entry.attrs[key] = val

    return entry


def parse_bench_info_line(line: str) -> Optional[dict]:
    """Parse ov_gpu_bench,info header line.

    Format: ov_gpu_bench,info,device=NAME,driver=VERSION,gpu_id=N
    Returns dict with 'device', 'driver', 'gpu_id' keys, or None.
    """
    line = line.strip()
    if not line.startswith("ov_gpu_bench,info,"):
        return None
    info = {}
    remainder = line[len("ov_gpu_bench,info,"):]
    for field in remainder.split(","):
        if "=" not in field:
            continue
        key, val = field.split("=", 1)
        info[key.strip()] = val.strip()
    return info


def parse_bench_verbose_log(lines: List[str]) -> Tuple[List[BenchVerboseEntry], dict]:
    """Parse multiple lines of bench verbose log.

    Returns (entries, info_dict) where info_dict contains device/driver info.
    """
    entries = []
    info = {}
    for line in lines:
        # Try info header first
        if not info:
            parsed_info = parse_bench_info_line(line)
            if parsed_info:
                info = parsed_info
                continue
        entry = parse_bench_verbose_line(line)
        if entry:
            entries.append(entry)
    return entries, info


# ============================================================================
# bench_kernel command generator
# ============================================================================

# Map OV primitive types to bench_kernel primitive flags
TYPE_TO_PRIMITIVE = {
    "fully_connected": "fc",
    "gemm": "gemm",
    "convolution": "conv",
    "scaled_dot_product_attention": "sdpa",
    "eltwise": "eltwise",
    "activation": "activation",
    "softmax": "softmax",
    "reduce": "reduce",
    "pooling": "pooling",
    "mvn": "mvn",
    "permute": "permute",
    "reorder": "reorder",
    "concatenation": "concatenation",
    "rms": "rms",
    "swiglu": "swiglu_standalone",
    "gather": "gather",
    "crop": "crop",
    "rope": "rope",
    "strided_slice": "strided_slice",
    "broadcast": "broadcast",
    "select": "select",
    "scatter_update": "scatter_update",
    "tile": "tile",
    "normalize": "normalize",
    "gather_elements": "gather_elements",
    "scatter_nd_update": "scatter_nd_update",
    "scatter_elements_update": "scatter_elements_update",
    "group_normalization": "group_normalization",
    "quantize": "quantize",
    "deconvolution": "deconv",
    "resample": "resample",
    "adaptive_pooling": "adaptive_pooling",
    "arg_max_min": "arg_max_min",
    "col2im": "col2im",
    "detection_output": "detection_output",
}

# Map fused op names to bench_kernel --attr-post-ops values
FUSED_OP_MAP = {
    "activation_relu": "relu",
    "activation_relu_negative_slope": "relu",
    "activation_gelu": "gelu_erf",
    "activation_gelu_tanh": "gelu_tanh",
    "activation_sigmoid": "sigmoid",
    "activation_tanh": "tanh",
    "activation_swish": "swish",
    "activation_hswish": "hardswish",
    "activation_mish": "mish",
    "activation_abs": "abs",
    "activation_sqrt": "sqrt",
    "activation_square": "square",
    "activation_elu": "elu",
    "activation_clamp": "clamp",
    "activation_exp": "exp",
    "activation_log": "log",
    "activation_negative": "negative",
    "activation_softrelu": "softrelu",
    "activation_softplus": "softplus",
    "activation_round": "round",
    "activation_hsigmoid": "hardsigmoid",
    "activation_hard_sigmoid": "hardsigmoid",
    "activation_linear": "linear",
    "activation_floor": "relu",
    "activation_ceil": "relu",
    "activation_log2": "relu",
    "activation_pow": "relu",
    "activation_reciprocal": "relu",
    "activation_selu": "relu",
    "activation_sign": "relu",
    "activation_sin": "relu",
    "activation_cos": "relu",
    "activation_softsign": "relu",
    "activation_tan": "relu",
    "activation_erf": "relu",
    "activation_none": "",
    "activation_act_unknown": "",  # Unknown activation type in verbose log → skip
    "eltwise_sum": "sum",
    "eltwise_prod": "prod",
    "eltwise_sub": "sub",
    "eltwise_div": "div",
    "eltwise_max": "max",
    "eltwise_min": "min",
    "eltwise_pow": "pow",
    "quantize": "quantize",
    # reorder is a type-conversion fused op, not a computational post-op → skip
    "reorder": "",
}


def dims_to_str(dims: List[int]) -> str:
    """Convert dimension list to 'AxBxC' string."""
    return "x".join(str(d) if d >= 0 else "?" for d in dims)


def _trim_trailing_ones(dims: List[int], min_rank: int = 1) -> List[int]:
    """Remove trailing dimensions that are 1, keeping at least min_rank dims.

    GPU verbose logs pad shapes to 4D with trailing 1s. For primitives like
    gather/scatter_nd, this inflates the output rank and can cause kernel selection
    failures. This strips the artificial padding.
    """
    result = list(dims)
    while len(result) > min_rank and result[-1] == 1:
        result.pop()
    return result


def _fix_permute_order(entry: BenchVerboseEntry) -> None:
    """Pad permute_order to match input shape rank.

    GPU verbose logs may emit a shorter permute order than the padded 4D shape.
    For example, order=1:2:0 (3D) but shape=4x4x4x1 (4D). Pad the order with
    the missing trailing dimension indices so reference and GPU see the same thing.
    """
    if "permute_order" not in entry.attrs:
        return
    order_str = entry.attrs["permute_order"]
    order = [int(x) for x in order_str.split(":")]
    # Get input shape rank
    if not entry.inputs:
        return
    shape_rank = len(entry.inputs[0][1])
    if len(order) < shape_rank:
        # Append missing indices (they are identity for the padded dims)
        for i in range(len(order), shape_rank):
            order.append(i)
        entry.attrs["permute_order"] = ":".join(str(x) for x in order)


def _fix_pooling_rounding_type(entry: BenchVerboseEntry) -> None:
    """Fix rounding_type for pooling entries.

    The GPU plugin's legacy path (with_output_size=true) hardcodes rounding_type=CEIL
    in the cldnn::pooling primitive even when the model uses FLOOR rounding. This causes
    the verbose log to report incorrect rounding_type=1. We correct this by comparing the
    actual output size (from out0=) with floor/ceil computed output sizes and choosing the
    matching rounding mode.
    """
    if entry.prim_type != "pooling" or "rounding_type" not in entry.attrs:
        return
    if not entry.inputs or not entry.outputs:
        return

    rounding_type = int(entry.attrs.get("rounding_type", "0"))
    if rounding_type == 0:  # FLOOR - no issue
        return

    in_dims = entry.inputs[0][1]
    out_dims = entry.outputs[0][1]

    kernel_str = entry.attrs.get("kernel", "")
    stride_str = entry.attrs.get("pool_strides", "")
    pb_str = entry.attrs.get("pads_begin", "")
    pe_str = entry.attrs.get("pads_end", "")

    if not kernel_str or not stride_str:
        return

    kernel = [int(x) for x in kernel_str.split("x")]
    stride = [int(x) for x in stride_str.split("x")]
    pb = [int(x) for x in pb_str.split("x")] if pb_str else [0] * len(kernel)
    pe = [int(x) for x in pe_str.split("x")] if pe_str else [0] * len(kernel)

    spatial_dims = len(kernel)
    if len(in_dims) < 2 + spatial_dims or len(out_dims) < 2 + spatial_dims:
        return

    # Check if the actual output matches floor or ceil formula
    import math
    matches_floor = True
    matches_ceil = True
    for i in range(spatial_dims):
        inp = in_dims[2 + i]
        out_actual = out_dims[2 + i]
        padded = inp + pb[i] + pe[i]
        floor_out = (padded - kernel[i]) // stride[i] + 1
        ceil_out = -(-((padded - kernel[i])) // stride[i]) + 1  # ceil division without import

        if floor_out != out_actual:
            matches_floor = False
        if ceil_out != out_actual:
            matches_ceil = False

    if matches_floor and not matches_ceil:
        # Output matches floor, not ceil - fix the rounding_type
        entry.attrs["rounding_type"] = "0"


def entry_to_bench_cmd(entry: BenchVerboseEntry, device: int = 0) -> Optional[str]:
    """Convert a BenchVerboseEntry to a bench_kernel command line string.

    Returns a command string for the batch file, or None if the primitive type
    is not supported.
    """
    # Fix known verbose log inaccuracies before conversion
    _fix_pooling_rounding_type(entry)
    _fix_permute_order(entry)

    primitive = TYPE_TO_PRIMITIVE.get(entry.prim_type)
    if primitive is None:
        return None

    def _append_shape(shape_list: List[str], dims: List[int]) -> None:
        s = dims_to_str(dims)
        if s:
            shape_list.append(s)

    # Build the command parts
    parts = []

    # Data types: preserve all logged inputs, then append output dtype.
    dts = [dt for dt, _dims, *_rest in entry.inputs]
    if entry.outputs:
        dts.append(entry.outputs[0][0])
    if dts:
        parts.append(f"--dt={':'.join(dts)}")

    # Shapes: preserve all logged input shapes for every primitive.
    # Keep primitive-specific normalization where historically needed.
    shapes = []
    for _, dims, *_rest in entry.inputs:
        normalized_dims = dims
        if primitive in ("fc", "gemm"):
            normalized_dims = _trim_trailing_ones(dims, min_rank=2)
        elif primitive in ("gather", "gather_elements"):
            normalized_dims = _trim_trailing_ones(dims)
        elif primitive == "detection_output":
            normalized_dims = _trim_trailing_ones(dims, min_rank=2)
        _append_shape(shapes, normalized_dims)

    # Some primitives require target/output shape in addition to all inputs.
    if primitive in ("crop", "strided_slice", "broadcast", "select") and entry.outputs:
        out_dims = entry.outputs[0][1]
        if primitive == "strided_slice":
            out_dims = _trim_trailing_ones(out_dims)
        _append_shape(shapes, out_dims)

    if any(not s for s in shapes):
        return None

    REQUIRED_SHAPES = {
        "fc": 2,
        "gemm": 2,
        "conv": 2,
        "deconv": 2,
        "sdpa": 3,
        "eltwise": 2,
        "gather": 2,
        "crop": 2,
        "rope": 3,
        "strided_slice": 2,
        "broadcast": 2,
        "select": 1,
        "scatter_update": 3,
        "scatter_nd_update": 3,
        "scatter_elements_update": 3,
        "gather_elements": 2,
        "quantize": 5,
        "detection_output": 3,
        "activation": 1,
    }
    if primitive in REQUIRED_SHAPES and len(shapes) < REQUIRED_SHAPES[primitive]:
        return None

    if shapes:
        parts.append(f"--shapes={':'.join(shapes)}")

    # Impl type
    if entry.impl and entry.impl not in ("unknown", "common", "cpu"):
        parts.append(f"--impl={entry.impl}")

    # Post-ops from fused operations (only for kernels that support them)
    PRIMITIVES_WITH_POST_OPS = {"fc", "gemm", "conv", "sdpa", "eltwise", "pooling", "activation"}
    post_ops = []
    if primitive in PRIMITIVES_WITH_POST_OPS:
        # Determine quantize output dtype for fused quantize ops.
        # Priority: output dtype > input dtype > default u8
        _INT_DTYPES = {"i8", "u8", "i4", "u4", "i32", "i64"}
        quant_dt = None
        if entry.outputs and entry.outputs[0][0] in _INT_DTYPES:
            quant_dt = entry.outputs[0][0]
        elif entry.inputs and entry.inputs[0][0] in _INT_DTYPES:
            quant_dt = entry.inputs[0][0]

        for fop in entry.fused_ops:
            # Split fused op into base name and optional params
            # New format: activation_swish:1, eltwise_sum:f16, etc.
            fop_parts = fop.split(":")
            base_name = fop_parts[0]
            params = fop_parts[1:]  # alpha, beta for activation; dt for eltwise

            mapped = FUSED_OP_MAP.get(base_name, base_name)
            if mapped:
                if base_name.startswith("activation_") and params:
                    # Pass through alpha[:beta] → e.g. "swish:1" or "linear:0.5:0.3"
                    mapped = mapped + ":" + ":".join(params)
                elif base_name.startswith("eltwise_") and params:
                    # Pass through eltwise dt → e.g. "sum:f16"
                    mapped = mapped + ":" + ":".join(params)
                elif base_name == "quantize" and quant_dt:
                    # Append inferred integer dtype → e.g. "quantize:i8"
                    mapped = mapped + ":" + quant_dt
                post_ops.append(mapped)
    if post_ops:
        parts.append(f"--attr-post-ops={'+'.join(post_ops)}")

    # Reorder-specific: truncate flag for Convert ops
    if primitive == "reorder" and entry.truncate:
        parts.append("--truncate")

    # Primitive-specific attributes from verbose log
    # These are passed through as --key=value flags to bench_kernel
    ATTR_FLAG_MAP = {
        # gemm
        "transpose_a": "--transpose_a",
        "transpose_b": "--transpose_b",
        "order0": "--gemm_order0",
        "order1": "--gemm_order1",
        "order_out_gemm": "--gemm_order_out",
        # sdpa
        "is_causal": "--is_causal",
        "order_q": "--order_q",
        "order_k": "--order_k",
        "order_v": "--order_v",
        "order_out": "--order_out",
        "scale_val": "--scale_val",
        # convolution
        "groups": "--groups",
        "strides": "--strides",
        "dilations": "--dilations",
        "padding_begin": "--padding_begin",
        "padding_end": "--padding_end",
        "grouped_weights_shape": "--grouped_weights_shape",
        # pooling
        "pool_mode": "--pool_mode",
        "kernel": "--kernel",
        "pool_strides": "--pool_strides",
        "pads_begin": "--pads_begin",
        "pads_end": "--pads_end",
        "rounding_type": "--rounding_type",
        # reduce
        "reduce_mode": "--reduce_mode",
        "keep_dims": "--keep_dims",
        "reduce_axes": "--reduce_axes",
        # softmax
        "axis": "--axis",
        # mvn
        "normalize_variance": "--normalize_variance",
        "epsilon": "--epsilon",
        "eps_inside_sqrt": "--eps_inside_sqrt",
        # eltwise
        "eltwise_mode": "--eltwise_mode",
        "pythondiv": "--pythondiv",
        # swiglu
        "glu_type": "--glu_type",
        "split_axis": "--split_axis",
        "split_length": "--split_length",
        "gate_idx": "--gate_idx",
        # gather
        "gather_axis": "--gather_axis",
        "batch_dim": "--batch_dim",
        "support_neg_ind": "--support_neg_ind",
        # rope
        "head_cnt": "--head_cnt",
        "head_size": "--head_size",
        "rotary_ndims": "--rotary_ndims",
        "is_interleaved": "--is_interleaved",
        "is_chatglm": "--is_chatglm",
        "is_qwen": "--is_qwen",
        "input_trans0213": "--input_trans0213",
        "slice_start": "--slice_start",
        "slice_stop": "--slice_stop",
        "gather_rank": "--gather_rank",
        # crop
        "offsets": "--offsets",
        # strided_slice
        "ss_begin": "--ss_begin",
        "ss_end": "--ss_end",
        "ss_strides": "--ss_strides",
        "begin_mask": "--begin_mask",
        "end_mask": "--end_mask",
        "shrink_axis_mask": "--shrink_axis_mask",
        "new_axis_mask": "--new_axis_mask",
        # concatenation
        "concat_axis": "--concat_axis",
        # tile
        "tile_repeats": "--tile_repeats",
        # normalize
        "across_spatial": "--across_spatial",
        # group_normalization
        "num_groups": "--num_groups",
        # quantize
        "levels": "--levels",
        # scatter_nd_update
        "indices_rank": "--indices_rank",
        # resample
        "resample_sizes": "--resample_sizes",
        "resample_mode": "--resample_mode",
        # permute
        "permute_order": "--permute_order",
        # broadcast
        "broadcast_axes": "--broadcast_axes",
        "broadcast_target": "--broadcast_target",
        # adaptive_pooling
        "adaptive_pool_mode": "--adaptive_pool_mode",
        "adaptive_pool_out": "--adaptive_pool_out",
        # arg_max_min
        "topk_mode": "--topk_mode",
        "top_k": "--top_k",
        # col2im
        "col2im_output_shape": "--col2im_output_shape",
        "col2im_kernel_shape": "--col2im_kernel_shape",
        # detection_output
        "det_num_classes": "--det_num_classes",
        "det_keep_top_k": "--det_keep_top_k",
        "det_top_k": "--det_top_k",
        "det_nms_threshold": "--det_nms_threshold",
        "det_confidence_threshold": "--det_confidence_threshold",
        "det_code_type": "--det_code_type",
        "det_share_location": "--det_share_location",
        "det_background_label_id": "--det_background_label_id",
        "det_variance_encoded": "--det_variance_encoded",
    }
    for attr_key, flag in ATTR_FLAG_MAP.items():
        if attr_key in entry.attrs and entry.attrs[attr_key] != "":
            parts.append(f"{flag}={entry.attrs[attr_key]}")

    if primitive == "scatter_nd_update" and "indices_rank" not in entry.attrs and len(entry.inputs) > 1:
        indices_rank = len(entry.inputs[1][1])
        if indices_rank > 0:
            parts.append(f"--indices_rank={indices_rank}")

    if primitive == "strided_slice":
        ss_begin = entry.attrs.get("ss_begin", "")
        new_axis_mask = entry.attrs.get("new_axis_mask", "")
        if ss_begin and not new_axis_mask and entry.inputs:
            ss_axes = ss_begin.count(":") + 1
            input_rank = len(entry.inputs[0][1])
            if ss_axes > input_rank:
                inferred = ["1"] * (ss_axes - input_rank) + ["0"] * input_rank
                parts.append(f"--new_axis_mask={':'.join(inferred)}")

    # Layout formats (from verbose log format field: dt:shape:format)
    in_fmts = [inp[2] for inp in entry.inputs if len(inp) > 2 and inp[2]]
    out_fmts = [out[2] for out in entry.outputs if len(out) > 2 and out[2]]
    if in_fmts:
        parts.append(f"--in_layouts={','.join(in_fmts)}")
    if out_fmts:
        parts.append(f"--out_layouts={','.join(out_fmts)}")

    # Construct the full command line for batch file
    # batch file format: kernel_flag args (one per line)
    cmd = f"--{primitive} " + " ".join(parts)
    return cmd


def entry_to_comment(entry: BenchVerboseEntry) -> str:
    """Generate a comment line for the entry."""
    time_str = f" {entry.time_us:.1f}us" if entry.time_us > 0 else ""
    return f"# {entry.prim_id} [{entry.impl}] {entry.kernel}{time_str}"


# ============================================================================
# Model shape generators for common architectures
# ============================================================================

def generate_llm_fc_batch(model_dim: int = 4096, ffn_dim: int = 11008,
                          vocab_size: int = 32000, batch_sizes: List[int] = None) -> List[str]:
    """Generate FC benchmark batch for LLM-like architecture."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 16, 32]

    lines = [
        "# LLM FC benchmark shapes",
        f"# model_dim={model_dim}, ffn_dim={ffn_dim}, vocab_size={vocab_size}",
        "#"
    ]

    for bs in batch_sizes:
        lines.append(f"# Batch={bs}")
        # QKV projection: input=[B, model_dim], weight=[3*model_dim, model_dim]
        lines.append(f"{bs}x{model_dim}:{3*model_dim}x{model_dim}")
        # Output projection: input=[B, model_dim], weight=[model_dim, model_dim]
        lines.append(f"{bs}x{model_dim}:{model_dim}x{model_dim}")
        # FFN up: input=[B, model_dim], weight=[ffn_dim, model_dim]
        lines.append(f"{bs}x{model_dim}:{ffn_dim}x{model_dim}")
        # FFN down: input=[B, ffn_dim], weight=[model_dim, ffn_dim]
        lines.append(f"{bs}x{ffn_dim}:{model_dim}x{ffn_dim}")
        # LM head: input=[B, model_dim], weight=[vocab_size, model_dim]
        lines.append(f"{bs}x{model_dim}:{vocab_size}x{model_dim}")
        lines.append("")

    return lines


def generate_sdpa_batch(num_heads: int = 32, head_dim: int = 128,
                        seq_lengths: List[int] = None, batch: int = 1) -> List[str]:
    """Generate SDPA benchmark batch for attention workloads."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]

    lines = [
        "# SDPA benchmark shapes",
        f"# num_heads={num_heads}, head_dim={head_dim}, batch={batch}",
        "#"
    ]

    for seq_len in seq_lengths:
        # Decode (single token query against full KV cache)
        lines.append(f"# seq_len={seq_len} decode")
        lines.append(f"{batch}x{num_heads}x1x{head_dim}:{batch}x{num_heads}x{seq_len}x{head_dim}:{batch}x{num_heads}x{seq_len}x{head_dim}")
        # Prefill
        lines.append(f"# seq_len={seq_len} prefill")
        lines.append(f"{batch}x{num_heads}x{seq_len}x{head_dim}:{batch}x{num_heads}x{seq_len}x{head_dim}:{batch}x{num_heads}x{seq_len}x{head_dim}")
        lines.append("")

    return lines


def generate_conv_batch() -> List[str]:
    """Generate Conv benchmark batch for common CNN shapes."""
    lines = [
        "# Convolution benchmark shapes (ResNet-50 like)",
        "#"
    ]

    shapes = [
        ("Conv1 7x7 s2", "1x3x224x224:64x3x7x7"),
        ("Conv2 3x3", "1x64x56x56:64x64x3x3"),
        ("Conv3 3x3", "1x128x28x28:128x128x3x3"),
        ("Conv4 3x3", "1x256x14x14:256x256x3x3"),
        ("Conv5 3x3", "1x512x7x7:512x512x3x3"),
        ("1x1 reduction", "1x256x56x56:64x256x1x1"),
        ("1x1 expansion", "1x64x56x56:256x64x1x1"),
    ]

    for desc, shape in shapes:
        lines.append(f"# {desc}")
        lines.append(shape)

    return lines


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="bench_kernel_converter - Convert OV GPU verbose logs to bench_kernel commands"
    )
    parser.add_argument("--input", "-i", help="Input verbose log file (use - for stdin)")
    parser.add_argument("--output", "-o", help="Output file for bench_kernel commands (default: stdout)")
    parser.add_argument("--mode", "-m", choices=["c", "p", "f", "cp", "r"], default="p",
                        help="Benchmark mode: c(correctness), p(perf), f(fast perf), cp(corr+perf), r(run 1iter) (default: p)")
    parser.add_argument("--kernel", "-k",
                        help="Filter: only include specified kernel type "
                             "(e.g. fc, conv, gemm, sdpa, eltwise, softmax, pooling, etc.)")
    parser.add_argument("--uniq", "-u", action="store_true",
                        help="Remove duplicate kernel configurations (like uniq)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Output args-only batch format (default: full command lines)")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="Override GPU device index (default: from log)")
    parser.add_argument("--impl", default="", help="Override implementation type: ocl, onednn (default: from log)")
    parser.add_argument("--dt", default="", help="Override data type(s) (default: from log)")
    # parser.add_argument("--generate", choices=["llm_fc", "sdpa", "conv"],
    #                     help="Generate batch file for common workload")
    # parser.add_argument("--model-dim", type=int, default=4096, help="Model dimension for LLM generation")
    # parser.add_argument("--ffn-dim", type=int, default=11008, help="FFN dimension for LLM generation")
    # parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size for LLM generation")
    # parser.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    # parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary statistics of parsed entries")

    args = parser.parse_args()

    lines = []

    if hasattr(args, 'generate') and args.generate:
        if args.generate == "llm_fc":
            lines = generate_llm_fc_batch(args.model_dim, args.ffn_dim, args.vocab_size)
        elif args.generate == "sdpa":
            lines = generate_sdpa_batch(args.num_heads, args.head_dim)
        elif args.generate == "conv":
            lines = generate_conv_batch()
    elif args.input:
        # Parse verbose log
        if args.input == '-':
            input_lines = sys.stdin.readlines()
        else:
            with open(args.input) as f:
                input_lines = f.readlines()

        # Filter for ov_gpu_bench lines
        bench_lines = [l for l in input_lines if "ov_gpu_bench" in l]
        entries, log_info = parse_bench_verbose_log(bench_lines)

        # Use device from log if not overridden
        if log_info:
            if log_info.get('device'):
                print(f"Log device: {log_info['device']}", file=sys.stderr)
            if log_info.get('driver'):
                print(f"Log driver: {log_info['driver']}", file=sys.stderr)

        if not entries:
            print("Warning: No ov_gpu_bench entries found. Ensure OV_GPU_BenchVerbose=1 is set.",
                  file=sys.stderr)
            return

        # Print summary if requested
        if args.summary:
            type_counts = {}
            for e in entries:
                type_counts[e.prim_type] = type_counts.get(e.prim_type, 0) + 1
            print("=== Verbose Log Summary ===", file=sys.stderr)
            print(f"Total entries: {len(entries)}", file=sys.stderr)
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                mapped = TYPE_TO_PRIMITIVE.get(t, "SKIP")
                print(f"  {t}: {c} -> {mapped}", file=sys.stderr)
            print("===========================", file=sys.stderr)

        # Filter by kernel type if requested
        if args.kernel:
            filtered = []
            for e in entries:
                mapped = TYPE_TO_PRIMITIVE.get(e.prim_type)
                if mapped == args.kernel:
                    filtered.append(e)
            entries = filtered

        # Filter out unsupported types and concat (verbose log doesn't capture multi-input concat)
        entries = [e for e in entries
                   if TYPE_TO_PRIMITIVE.get(e.prim_type) is not None
                   and e.prim_type != "concatenation"]

        # Deduplication
        if args.uniq:
            seen = set()
            deduped = []
            for e in entries:
                # Create a key from type + dtypes + shapes
                in_key = tuple((dt, tuple(dims), fmt) for dt, dims, fmt in e.inputs)
                out_key = tuple((dt, tuple(dims), fmt) for dt, dims, fmt in e.outputs)
                key = (e.prim_type, e.impl, in_key, out_key, tuple(e.fused_ops))
                if key not in seen:
                    seen.add(key)
                    deduped.append(e)
            entries = deduped

        # Resolve device: CLI override > entry device field > log info > default 0
        if args.device is not None:
            device_id = args.device
        elif entries and entries[0].device:
            # Parse from "gpu.N" format
            m = re.match(r'gpu\.(\d+)', entries[0].device)
            device_id = int(m.group(1)) if m else 0
        elif 'gpu_id' in log_info:
            device_id = int(log_info['gpu_id'])
        else:
            device_id = 0


        # Generate commands
        header_lines = ["# Generated from OV GPU bench verbose log"]
        if log_info.get('device'):
            header_lines.append(f"# Device: {log_info['device']}")
        if log_info.get('driver'):
            header_lines.append(f"# Driver: {log_info['driver']}")
        if log_info.get('gpu_id'):
            header_lines.append(f"# GPU ID: {log_info['gpu_id']}")
        header_lines.append(f"# Mode: {args.mode}")
        header_lines.append("#")
        lines = header_lines
        for entry in entries:
            comment = entry_to_comment(entry)
            cmd = entry_to_bench_cmd(entry, device_id)
            if cmd:
                # Add mode to each command
                cmd += f" --mode={args.mode}"
                # Apply overrides
                if args.dt:
                    cmd = re.sub(r'--dt=\S+', f'--dt={args.dt}', cmd)
                if args.impl:
                    cmd = re.sub(r'--impl=\S+', f'--impl={args.impl}', cmd)
                    if '--impl=' not in cmd:
                        cmd += f' --impl={args.impl}'

                lines.append(comment)
                if args.batch:
                    lines.append(cmd)
                else:
                    lines.append(f"ov_gpu_bench_kernel {cmd} --device={device_id}")
    else:
        parser.print_help()
        return

    # Output
    output_text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Written {len([l for l in lines if not l.startswith('#')])} commands to {args.output}",
              file=sys.stderr)
    else:
        print(output_text)


if __name__ == "__main__":
    main()
