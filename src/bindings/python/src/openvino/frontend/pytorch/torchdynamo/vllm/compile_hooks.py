# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM compile-time hooks called from torchdynamo.compile; each no-ops on graphs without its marker."""

import logging
import os

from openvino.frontend.pytorch.torchdynamo.vllm import preset as _preset

logger = logging.getLogger(__name__)


def decoder_default_float_type(options):
    """The dtype vLLM built the model with (options["default_float_type"]), or None without "canonical_precision"."""
    if not _preset.bool_opt(options, "canonical_precision", False):
        return None
    return options.get("default_float_type") if options else None


def apply_post_convert(om, options):
    """Run vLLM hooks on the freshly converted Model, before serialization or input shaping."""
    if _preset.is_vllm_preset(options):
        # Read back in CommonOptimizations to gate NormalizeVLLMRoPE.
        om.set_rt_info(True, "vllm_model")
    register_pa_parameters(om)
    normalize_concat_ranks(om)


def apply_input_shapes(om, args, options, gm):
    """Resolve Python-int FX inputs and shape the tensor Parameters; False leaves that to the caller."""
    if not (_preset.is_vllm_preset(options) or any(isinstance(a, int) for a in args)):
        return False
    bake_symint_constants(
        om, args, dyn_shapes=_preset.bool_opt(options, "dynamic_shapes", True), gm=gm)
    return True


def _vllm_omp_core_ids():
    """The cores vLLM's OMPProcessManager assigned this worker (from its OpenMP env vars), or None."""
    gomp = os.environ.get("GOMP_CPU_AFFINITY")
    if gomp:
        return {int(c) for c in gomp.split()}
    places = os.environ.get("OMP_PLACES")
    if places:
        return {int(c) for c in places.strip("{").strip("}").split(",") if c}
    kmp = os.environ.get("KMP_AFFINITY")
    if kmp and "proclist=[" in kmp:
        proclist = kmp.split("proclist=[", 1)[1].split("]", 1)[0]
        return {int(c) for c in proclist.split(",") if c}
    return None


def widen_affinity_if_needed():
    """Widen the worker's affinity (narrowed to one CPU) to vLLM's assigned cores, else all CPUs.

    TBB samples the affinity on first use, so a 1-CPU mask before core.compile would pin OV to one thread.
    """
    try:
        cur = os.sched_getaffinity(0)
        req = int(os.environ.get("OV_INFERENCE_NUM_THREADS", "0") or 0)
        if req == 0 or len(cur) < req:
            target = _vllm_omp_core_ids() or set(range(os.cpu_count() or 1))
            os.sched_setaffinity(0, target)
    except Exception as _e:
        logger.debug("widen_affinity skipped: %s", _e)


def symint_shape_sources(gm, args):
    """Map int FX inputs to the tensor dimension carrying the same symbol: {int_arg_index: (tensor_arg_index, dim)}.

    Read off meta['val'] so two symbols that coincide on this trace aren't conflated.
    """
    if gm is None:
        return {}
    try:
        import torch
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        if len(placeholders) != len(args):
            return {}
        # symbol -> (tensor arg index, dim); the first carrier wins.
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
    """Replace integer FX input Parameters (their upper bound of 0 collapses shapes) by ShapeOf dims or Constants.

    Tensor Parameters go dynamic when dyn_shapes is set or every int came from a ShapeOf, which must not const-fold.
    """
    import torch
    import numpy as np
    from openvino import PartialShape, opset1 as _opset1, opset8 as _opset8
    from openvino.frontend.pytorch.utils import pt_to_ov_type_map

    sources = symint_shape_sources(gm, args)
    n_int_args = sum(1 for a in args if isinstance(a, int))

    # Match by name: the frontend can drop a consumer-less placeholder, shifting om.inputs.
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"] if gm is not None else None
    name_to_input = None
    if placeholders is not None and len(placeholders) == len(args):
        name_to_input = {}
        for port in om.inputs:
            for port_name in port.get_names():
                name_to_input.setdefault(port_name, port)

    def _om_input_for(idx):
        if name_to_input is not None:
            return name_to_input.get(placeholders[idx].name)
        return om.inputs[idx] if idx < len(om.inputs) else None

    params_to_remove = []
    # Two symints can share a source tensor's ShapeOf.
    shape_of_cache = {}
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            om_input = _om_input_for(idx)
            if om_input is None:
                continue  # dropped by the frontend
            param_node = om_input.get_node()
            src = sources.get(idx)
            if src is None:
                repl = _opset1.constant(np.array([int(input_data)], dtype=np.int64))
            else:
                tensor_arg_idx, dim = src
                shape_of = shape_of_cache.get(tensor_arg_idx)
                if shape_of is None:
                    tensor_input = _om_input_for(tensor_arg_idx)
                    shape_of = _opset8.shape_of(tensor_input, output_type="i64")
                    shape_of_cache[tensor_arg_idx] = shape_of
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

    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            continue
        om_input = _om_input_for(idx)
        if om_input is None:
            continue
        om_input.get_node().set_element_type(pt_to_ov_type_map[str(input_data.dtype)])
        # Size values arrive as 1-element tensors; keep them static or derived shapes get a dynamic rank.
        meta_val = None
        if placeholders is not None and len(placeholders) == len(args):
            meta_val = placeholders[idx].meta.get("val")
        if isinstance(meta_val, (torch.SymInt, int)):
            om_input.get_node().set_partial_shape(PartialShape(list(input_data.size())))
        elif dyn:
            om_input.get_node().set_partial_shape(PartialShape([-1] * input_data.ndim))
        else:
            om_input.get_node().set_partial_shape(PartialShape(list(input_data.size())))

    # The caller must re-infer downstream shapes, or a stale ShapeOf const-folds back to the frozen size.


def register_pa_parameters(om):
    """Register the ``__pa__`` side-channel Parameters the paged_attention translator emits as model inputs."""
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
    """Bypass Unsqueezes of rank>=1 values feeding a Concat whose inputs disagree in static rank."""
    def _rank_ge_1(output):
        node = output.get_node()
        ps = output.get_partial_shape()
        if ps.rank.is_static and ps.rank.get_length() >= 1:
            return True
        if node.get_type_name() == "Constant":
            return len(node.get_output_shape(0)) >= 1
        return False

    def _find_targets(concats):
        for node in concats:
            ranks = set()
            for i in range(node.get_input_size()):
                ps = node.input_value(i).get_partial_shape()
                if ps.rank.is_static:
                    ranks.add(ps.rank.get_length())
            if len(ranks) < 2:
                continue
            for i in range(node.get_input_size()):
                src = node.input_value(i)
                src_node = src.get_node()
                if src_node.get_type_name() != "Unsqueeze":
                    continue
                inner = src_node.input_value(0)
                if _rank_ge_1(inner):
                    yield node, i, inner

    try:
        # Rewiring never adds or removes Concats.
        concats = [node for node in om.get_ordered_ops()
                   if node.get_type_name() == "Concat" and node.get_input_size() >= 2]
        for _ in range(64):
            targets = list(_find_targets(concats))
            if not targets:
                break
            for node, i, inner in targets:
                node.input(i).replace_source_output(inner)
        om.validate_nodes_and_infer_types()
    except Exception as e:
        logger.debug("concat-rank normalization skipped: %s", e)


def model_float_precision(om):
    """The model's float dtype ("bf16"/"f16") or None: from rt_info, else the key_cache port, else any I/O port."""
    if om is None:
        return None
    if om.has_rt_info("default_float_type"):
        name = om.get_rt_info("default_float_type").astype(str)
        return name if name in _preset.SUPPORTED_FLOAT_PRECISIONS else None

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
    """Tensor names plus the node's friendly name (the frontend does not always set the former)."""
    names = set(port.get_names())
    names.add(port.get_node().get_friendly_name())
    return names


def _is_kv_cache_port(port, field=("key_cache", "value_cache")):
    """True if `port` is a PagedAttention KV-cache Parameter (names carry a layer-index suffix)."""
    return any(n.startswith("__pa__") and any(f in n for f in field)
               for n in _port_names(port))


def retype_kv_cache_parameters(om, et_name):
    """Retype the KV-cache Parameters to `et_name`; return how many changed.

    Otherwise the plugin splices a Convert after them and PA's in-place cache writes land in its temporary.
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


def apply_post_config(config, device, options, om):
    """Fill the vLLM KV-cache and FC-quantization defaults into a CPU compile config; caller entries win."""
    if device != "CPU" or not _preset.is_vllm_preset(options):
        return
    for key, value in _preset._PRESET_CONFIG.items():
        config.setdefault(key, value)

    precisions = _preset.precision_config(model_float_precision(om))
    # Env var overrides; setting only one risks a mismatch.
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

    retype_kv_cache_parameters(om, config["KV_CACHE_PRECISION"])
