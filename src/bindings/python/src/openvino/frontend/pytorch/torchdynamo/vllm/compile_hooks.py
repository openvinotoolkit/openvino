# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific compile-time hooks.

Called from torchdynamo.compile.openvino_compile to keep the generic compile
path free of vLLM knowledge. Each hook no-ops on graphs lacking its marker (a
__pa__ Parameter prefix, a vLLM Concat pattern).
"""

import logging
import os

from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset

logger = logging.getLogger(__name__)


# The four apply_* entry points below are grouped so compile.py needs one
# try/except per call site rather than one per hook.

def apply_post_convert(om, options):
    """Run vLLM hooks on the freshly-converted Model.

    Runs right after ``fe.convert(im)`` and before serialization or input
    shaping.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if bool_opt(options, "vllm", False):
        # Read back in CommonOptimizations/CPU PostLpt to gate vLLM-only passes.
        om.set_rt_info(True, "vllm_model")
    register_pa_parameters(om)
    normalize_concat_ranks(om)
    if bool_opt(options, "fc_decompress", True):
        rewrite_fc_decompression(om)


def apply_input_shapes(om, args, options, gm=None):
    """Resolve Python-int FX inputs and shape the remaining tensor Parameters.

    ``gm`` is optional for older callers, but without it int inputs can only be
    frozen at trace-time values (see bake_symint_constants).

    Returns False when the caller should fall through to its own loop: no int
    args and no vLLM preset.
    """
    from openvino.frontend.pytorch.torchdynamo.vllm.preset import bool_opt
    if not (bool_opt(options, "vllm", False) or any(isinstance(a, int) for a in args)):
        return False
    bake_symint_constants(
        om, args, dyn_shapes=bool_opt(options, "dynamic_shapes", True), gm=gm)
    return True


def apply_post_config(config, device, options, om=None):
    """Fill the vLLM defaults into the core.compile_model config.

    Called once the caller has built ``config`` and set CACHE_DIR. ``om`` is
    optional for older callers, but without it float precisions cannot be
    derived from the model and fall back to the preset default.
    """
    apply_kv_cache_config_defaults(config, device, options, om=om)


def widen_affinity_if_needed(options):
    """Widen process CPU affinity to all cores when narrower than needed.

    Needed when the mask is narrower than the requested OV thread count.
    vLLM's ``init_cpu_threads_env`` pins the worker to one CPU before
    ``torch.compile``, and TBB/OV sample affinity on first parallel use -- so a
    1-CPU mask locks ``INFERENCE_NUM_THREADS=1`` whatever config we pass.
    Widening before ``core.compile`` is what lets the pool inherit a useful
    mask at creation. No-op without sched_getaffinity, or if already wide
    enough.

    TODO: this also overrides a deliberate taskset/numactl pin -- revisit.
    """
    try:
        cur = os.sched_getaffinity(0)
        from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_config
        cfg = _get_config(options) or {}
        req = int(cfg.get("INFERENCE_NUM_THREADS",
                          os.environ.get("OV_INFERENCE_NUM_THREADS", "0")) or 0)
        if req == 0 or len(cur) < req:
            os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except Exception as _e:
        logger.debug("widen_affinity skipped: %s", _e)


def symint_shape_sources(gm, args):
    """Map int FX inputs to the tensor input dimension that carries them.

    Dynamo passes symbolic sizes as plain Python-int graph inputs alongside
    the tensor whose dimension they equal (e.g. num_tokens == input_ids.shape[0]).
    Returns {int_arg_index: (tensor_arg_index, dim)}, read off meta['val'] so
    two symbols that coincide on this trace aren't conflated. Empty when gm is
    None, placeholders don't line up with args, or the trace is static.
    """
    if gm is None:
        return {}
    try:
        import torch
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(args):
            return {}
        # symbol -> (tensor arg index, dim). First carrier wins; any tensor
        # carrying it is an equally valid source.
        symbol_src = {}
        for idx, node in enumerate(placeholders):
            meta_val = node.meta.get("val", None)
            if not isinstance(meta_val, torch.Tensor):
                continue
            for dim, extent in enumerate(meta_val.shape):
                if isinstance(extent, torch.SymInt):
                    symbol_src.setdefault(str(extent.node.expr), (idx, dim))
        sources = {}
        for idx, node in enumerate(placeholders):
            if not isinstance(args[idx], int):
                continue
            meta_val = node.meta.get("val", None)
            if not isinstance(meta_val, torch.SymInt):
                continue
            src = symbol_src.get(str(meta_val.node.expr))
            if src is not None:
                sources[idx] = src
        return sources
    except Exception as e:
        logger.debug("symint shape sourcing unavailable: %s", e)
        return {}


def bake_symint_constants(om, args, dyn_shapes: bool = True, gm=None):
    """Resolve integer FX inputs (seq_lens, past_lens, ...) and drop their Parameters.

    Left as OV Parameters, shape inference takes their unset upper bound of 0
    and collapses downstream ops to size 0. Each is replaced by
    Gather(ShapeOf(tensor_input), dim) when a source tensor dim is known (one
    compiled model then serves all shapes), else a frozen trace-time Constant.
    Tensor Parameters go fully dynamic when dyn_shapes is set or every int came
    from a ShapeOf, to avoid const-folding that ShapeOf back to a fixed size.
    """
    import torch
    import numpy as np
    from openvino import Type, PartialShape, opset1 as _opset1, opset8 as _opset8

    _dtype_mapping = {
        torch.float32: Type.f32, torch.float64: Type.f64,
        torch.float16: Type.f16, torch.bfloat16: Type.bf16,
        torch.int64: Type.i64,
        torch.int32: Type.i32, torch.uint8: Type.u8,
        torch.int8: Type.i8, torch.bool: Type.boolean,
    }

    sources = symint_shape_sources(gm, args)
    n_int_args = sum(1 for a in args if isinstance(a, int))

    params_to_remove = []
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            param_node = om.inputs[idx].get_node()
            src = sources.get(idx)
            if src is None:
                repl = _opset1.constant(np.array([int(input_data)], dtype=np.int64))
            else:
                # om.inputs is still 1:1 with args; Parameters are removed after.
                tensor_arg_idx, dim = src
                shape_of = _opset8.shape_of(om.inputs[tensor_arg_idx], output_type="i64")
                repl = _opset8.gather(
                    shape_of,
                    _opset1.constant(np.array([dim], dtype=np.int64)),
                    _opset1.constant(np.array(0, dtype=np.int64)))
            for consumer in list(param_node.output(0).get_target_inputs()):
                consumer.replace_source_output(repl.output(0))
            params_to_remove.append(param_node)
    for param in params_to_remove:
        om.remove_parameter(param)

    all_symints_sourced = n_int_args > 0 and len(sources) == n_int_args
    if all_symints_sourced and not dyn_shapes:
        logger.debug("symint inputs sourced from ShapeOf; forcing dynamic input shapes")
    dyn = dyn_shapes or all_symints_sourced

    tensor_idx = 0
    for input_data in args:
        if isinstance(input_data, int):
            continue
        om.inputs[tensor_idx].get_node().set_element_type(_dtype_mapping[input_data.dtype])
        if dyn:
            om.inputs[tensor_idx].get_node().set_partial_shape(
                PartialShape([-1] * input_data.ndim))
        else:
            om.inputs[tensor_idx].get_node().set_partial_shape(
                PartialShape(list(input_data.size())))
        tensor_idx += 1

    # set_partial_shape only touches the Parameter; the caller must re-infer
    # downstream shapes, or a stale ShapeOf const-folds back to the frozen size.


def register_pa_parameters(om):
    """Register dangling ``__pa__`` Parameters as model inputs.

    The paged_attention translator emits side-channel Parameters (KV cache,
    block tables, past_lens); unregistered, the Model fails validation with
    ``unregistered_parameters``.
    """
    try:
        existing_ids = {id(p) for p in om.get_parameters()}
        to_add = []
        for node in om.get_ordered_ops():
            if node.get_type_name() != "Parameter":
                continue
            if id(node) in existing_ids:
                continue
            if node.get_friendly_name().startswith("__pa__"):
                to_add.append(node)
        if to_add:
            om.add_parameters(to_add)
    except Exception as e:
        logger.debug("PA parameter registration skipped: %s", e)


def normalize_concat_ranks(om):
    """Strip redundant Unsqueeze wrappers feeding Concat.

    vLLM's symint-heavy graphs emit Unsqueeze wrappers that leave
    rank-mismatched Concat inputs on list-construct nodes. Bypasses each
    Unsqueeze whose inner input is already rank>=1, scoped to Concats whose
    inputs actually disagree in static rank -- not any validation failure,
    which may be unrelated. No-op on graphs that already pass it.
    """
    def _rank_ge_1(output):
        node = output.get_node()
        ps = output.get_partial_shape()
        if ps.rank.is_static and ps.rank.get_length() >= 1:
            return True
        if node.get_type_name() == "Constant":
            return len(node.get_output_shape(0)) >= 1
        return False

    def _find_targets():
        for node in om.get_ordered_ops():
            if node.get_type_name() != "Concat" or node.get_input_size() < 2:
                continue
            ranks = set()
            for i in range(node.get_input_size()):
                ps = node.input_value(i).get_partial_shape()
                if ps.rank.is_static:
                    ranks.add(ps.rank.get_length())
            if len(ranks) < 2:
                continue  # Inputs already rank-agree: not this pass's pattern.
            for i in range(node.get_input_size()):
                src = node.input_value(i)
                src_node = src.get_node()
                if src_node.get_type_name() != "Unsqueeze":
                    continue
                inner = src_node.input_value(0)
                if _rank_ge_1(inner):
                    yield node, i, inner

    try:
        for _ in range(64):
            targets = list(_find_targets())
            if not targets:
                break
            for node, i, inner in targets:
                node.input(i).replace_source_output(inner)
        om.validate_nodes_and_infer_types()
    except Exception as e:
        logger.debug("concat-rank normalization skipped: %s", e)


def model_float_precision(om):
    """Return the model's float dtype as an OV type name ("bf16"/"f16"), or None.

    Prefers the PagedAttention key_cache Parameter (compute precision must
    match it, see preset.precision_config); falls back to output/input element
    types on graphs with no PA op. Reads ports rather than walking Constants
    to stay O(#ports) on graphs with thousands of frozen weights.
    """
    if om is None:
        return None

    def narrow(element_type):
        name = element_type.get_type_name()
        return name if name in _preset.SUPPORTED_FLOAT_PRECISIONS else None

    try:
        for port in om.inputs:
            if _is_kv_cache_port(port, field=("key_cache",)):
                if (et := narrow(port.get_element_type())) is not None:
                    return et
        for port in list(om.outputs) + list(om.inputs):
            if (et := narrow(port.get_element_type())) is not None:
                return et
    except Exception as _e:
        logger.debug("model_float_precision failed, falling back to preset: %s", _e)
    return None


def _port_names(port):
    """Every name a port answers to.

    Tensor names plus the node's friendly name (the frontend sets the
    latter, not always the former).
    """
    names = set(port.get_names())
    names.add(port.get_node().get_friendly_name())
    return names


def _is_kv_cache_port(port, field=("key_cache", "value_cache")):
    """True if `port` is a PagedAttention KV-cache Parameter.

    Substring match, not suffix: the frontend appends the layer index,
    giving names like ``__pa__unknown_layer__key_cache_15``.
    """
    return any(n.startswith("__pa__") and any(f in n for f in field)
               for n in _port_names(port))


def retype_kv_cache_parameters(om, et_name):
    """Declare the PagedAttention key_cache/value_cache Parameters as `et_name`.

    If left at the model dtype while inference precision differs, the plugin
    splices a Convert after the Parameter -- fatal here, since PA writes K/V
    in place into its input buffer and the write would land in that Convert's
    temporary instead of the persistent cache. Returns the number changed.
    """
    from openvino import Type
    target = {"bf16": Type.bf16, "f16": Type.f16, "f32": Type.f32}.get(et_name)
    if om is None or target is None:
        return 0
    changed = 0
    try:
        for port in om.inputs:
            if _is_kv_cache_port(port) and port.get_element_type() != target:
                port.get_node().set_element_type(target)
                changed += 1
        if changed:
            om.validate_nodes_and_infer_types()
            logger.debug("retyped %d KV-cache Parameters to %s", changed, et_name)
    except Exception as _e:
        logger.debug("KV-cache Parameter retype to %s skipped: %s", et_name, _e)
    return changed


def apply_kv_cache_config_defaults(config, device, options=None, om=None):
    """Fill the vLLM KV-cache and FC-quantization defaults into the OV CPU config.

    Caller-supplied entries win. No-op on non-CPU devices and on non-vLLM
    callers, which compile.py also routes here.
    """
    if device != "CPU" or not _preset.is_vllm_preset(options):
        return
    for key, value in _preset._PRESET_CONFIG.items():
        config.setdefault(key, value)

    # Derived together from the model's float dtype: the CPU PA kernel only
    # exists for matching (compute, cache) pairs.
    precisions = _preset.precision_config(model_float_precision(om))
    # Env vars remain an escape hatch; setting only one risks a mismatch.
    for key, env in (("KV_CACHE_PRECISION", "OV_KV_CACHE_PRECISION"),
                     ("INFERENCE_PRECISION_HINT", "OV_INFERENCE_PRECISION_HINT")):
        if (override := os.environ.get(env)):
            config.setdefault(key, override)
        else:
            config.setdefault(key, precisions[key])
    if config["KV_CACHE_PRECISION"] != config["INFERENCE_PRECISION_HINT"]:
        logger.warning(
            "KV_CACHE_PRECISION=%s and INFERENCE_PRECISION_HINT=%s differ; the OV CPU "
            "PagedAttention kernel is only built for matching pairs and compile_model "
            "may reject this combination.",
            config["KV_CACHE_PRECISION"], config["INFERENCE_PRECISION_HINT"])

    # Make the graph agree with the config, or the plugin splices in a
    # Convert that silently drops every cache write.
    retype_kv_cache_parameters(om, config["KV_CACHE_PRECISION"])

    if "DYNAMIC_QUANTIZATION_GROUP_SIZE" not in config:
        # On-the-fly int8 FC activations: vnni int8 GEMM beats f32 GEMM by a
        # lot. Matches OV GenAI CPU behavior.
        config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(
            os.environ.get("DYNAMIC_QUANTIZATION_GROUP_SIZE", "32"))


def rewrite_fc_decompression(om):
    """Rewrite MatMul(X, Const_f16/bf16) into the oneDNN-BRGEMM-friendly form.

    Inserts a decompression Convert to f32 on the constant weight so
    ConvertMatMulToFC routes to brgemm_avx512_f32 instead of the slower
    gemm_mlas_f32 fallback; upcasts the activation to match and downcasts the
    output back. f32 weights and quantized paths are skipped.
    """
    from openvino import opset1 as _o1
    from openvino import Type
    try:
        for mm in list(om.get_ordered_ops()):
            if mm.get_type_name() != "MatMul":
                continue
            try:
                tb = mm.get_transpose_b()
            except Exception:
                continue
            if tb:
                continue  # already transpose_b=true
            src = mm.input_value(1).get_node()
            const = None
            new_tb = False
            if src.get_type_name() == "Transpose":
                inner = src.input_value(0).get_node()
                if inner.get_type_name() == "Constant":
                    perm_node = src.input_value(1).get_node()
                    if perm_node.get_type_name() == "Constant":
                        perm = list(perm_node.get_data().flatten())
                        if perm == [1, 0]:
                            const = inner
                            new_tb = True
            elif src.get_type_name() == "Constant":
                const = src
            if const is None:
                continue
            # The plugin's weight-decompression FC path takes inputType=f32 with
            # weightsType in {f16, bf16}; f32 needs no decompression.
            w_et = const.get_element_type()
            if w_et not in (Type.f16, Type.bf16):
                continue
            conv_w = _o1.convert(const.output(0), "f32")
            try:
                # Key name matters: is_decompression() looks for exactly
                # "decompression_0".
                conv_w.get_rt_info()["decompression_0"] = True
            except Exception:
                pass
            mm.input(1).replace_source_output(conv_w.output(0))
            # Upcast activation to f32; downcast each consumer of MatMul back.
            act_src = mm.input_value(0)
            act_et = act_src.get_element_type()
            if act_et in (Type.f16, Type.bf16):
                conv_a = _o1.convert(act_src, "f32")
                mm.input(0).replace_source_output(conv_a.output(0))
                out = mm.output(0)
                consumers = list(out.get_target_inputs())
                down = _o1.convert(out, act_et)
                for cin in consumers:
                    cin.replace_source_output(down.output(0))
            try:
                mm.set_transpose_b(new_tb if new_tb else mm.get_transpose_b())
            except Exception:
                pass
        om.validate_nodes_and_infer_types()
    except Exception as e:
        logger.debug("FC_DECOMPRESS rewrite failed: %s", e)
