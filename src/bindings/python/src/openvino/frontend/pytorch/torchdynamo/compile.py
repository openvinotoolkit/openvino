# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

import os
import logging
from hashlib import sha256

import torch
import torch.overrides
from torch.fx import GraphModule

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino import Core, Type, PartialShape, serialize
from openvino.frontend.pytorch.torchdynamo.backend_utils import (
    _get_cache_dir,
    _get_device,
    _get_config,
    _is_cache_dir_in_config,
    _bool_opt,
    _config_with_vllm_defaults,
)

logger = logging.getLogger(__name__)


def cached_model_name(model_hash_str, device, args, cache_root, reversed=False):  # noqa: VNE003
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        logger.warning("Cache directory %s cannot be created. Model caching is disabled. Error: %s", cache_root, error)
        return None

    inputs_str = ""
    for input_data in args:
        arg_str = str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
        if reversed:
            inputs_str = "_" + arg_str + inputs_str
        else:
            inputs_str += "_" + arg_str
    inputs_str = sha256(inputs_str.encode("utf-8")).hexdigest()
    file_name += inputs_str

    return file_name


def openvino_compile_cached_model(cached_model_path, options, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(example_inputs):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    config = {}

    if _is_cache_dir_in_config(options):
        config = _get_config(options)
    else:
        config["CACHE_DIR"] = _get_cache_dir(options)

    compiled_model = core.compile_model(om, _get_device(options), config)

    return compiled_model


def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None, options=None):
    # Some callers (e.g. vLLM's init_cpu_threads_env) pin the worker process
    # to a single CPU via sched_setaffinity before reaching here. TBB/OV sample
    # affinity on first parallel use, so a 1-CPU mask locks
    # INFERENCE_NUM_THREADS=1 regardless of what we request.
    #
    # Default OFF to avoid silently overriding a standalone caller's explicit
    # taskset/numactl pinning. Callers that need the widening (vLLM plugin)
    # pass options={"unbind_affinity": True} or options={"vllm": True}.
    # Even when enabled, only widen if the current affinity has fewer cores
    # than the requested thread count.
    if _bool_opt(options, "unbind_affinity", False):
        try:
            cur = os.sched_getaffinity(0)
            cfg = _get_config(options) or {}
            req = int(cfg.get("INFERENCE_NUM_THREADS",
                              os.environ.get("OV_INFERENCE_NUM_THREADS", "0")) or 0)
            if req == 0 or len(cur) < req:
                os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
        except Exception:
            pass
    core = Core()

    device = _get_device(options)
    cache_root = _get_cache_dir(options)
    file_name = cached_model_name(model_hash_str, device, args, cache_root)

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for input_data in args:
            if isinstance(input_data, int):
                input_types.append(torch.int64)
                input_shapes.append(torch.Size([1]))
            else:
                input_types.append(input_data.type())
                input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm)

        im = fe.load(decoder)

        om = fe.convert(im)

        # vLLM-specific compile hooks (register __pa__ Parameters, normalize
        # symint-heavy Concat ranks). Both are no-ops on standalone graphs.
        try:
            from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh
            _vh.register_pa_parameters(om)
            _vh.normalize_concat_ranks(om)
        except Exception as _ee:
            logger.debug("vllm.compile_hooks unavailable: %s", _ee)

        # Optional MatMul(X, Const_f16/bf16) rewrite to the oneDNN BRGEMM
        # decompression form. Lifts FP16/bf16 weight FCs onto the brgemm path.
        if _bool_opt(options, "fc_decompress", True):
            try:
                from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh_fc
                _vh_fc.rewrite_fc_decompression(om)
            except Exception as _ee:
                logger.debug("vllm.rewrite_fc_decompression skipped: %s", _ee)

        if file_name is not None:
            serialize(om, file_name + ".xml", file_name + ".bin")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    # Symint FX inputs are Python ints that torch.compile has already
    # specialized for this trace. Bake them as Constants so OV's shape
    # inference propagates concrete bounds through Broadcast/Reshape rather
    # than using the unset Parameter upper-bound of 0.
    from openvino import opset1 as _opset1
    import numpy as _np_compile
    _params_to_remove = []
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            _param_node = om.inputs[idx].get_node()
            _const = _opset1.constant(_np_compile.array([int(input_data)], dtype=_np_compile.int64))
            for _consumer in list(_param_node.output(0).get_target_inputs()):
                _consumer.replace_source_output(_const.output(0))
            _params_to_remove.append(_param_node)
    for _p in _params_to_remove:
        om.remove_parameter(_p)

    _tensor_idx = 0
    _dyn = _bool_opt(options, "dynamic_shapes", True)
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            continue  # Already baked as Constant above
        om.inputs[_tensor_idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        if _dyn:
            # Dynamic shapes so the compiled OV model is reused across
            # different batch sizes / seq lengths / past_lens rather than
            # recompiling per shape. Disable with options["dynamic_shapes"]=False.
            om.inputs[_tensor_idx].get_node().set_partial_shape(
                PartialShape([-1] * input_data.ndim))
        else:
            om.inputs[_tensor_idx].get_node().set_partial_shape(
                PartialShape(list(input_data.size())))
        _tensor_idx += 1

    om.validate_nodes_and_infer_types()

    config = _config_with_vllm_defaults(options)

    if model_hash_str is not None:
        if not _is_cache_dir_in_config(options):
            config["CACHE_DIR"] = cache_root

    # vLLM-specific OV-config defaults (KV cache precision, FC dynamic-
    # quantization group, narrow-float GEMM hint). No-op on non-CPU devices
    # and on standalone paths where the values are passed explicitly.
    try:
        from openvino.frontend.pytorch.torchdynamo.vllm import compile_hooks as _vh
        _vh.apply_kv_cache_config_defaults(config, device, options)
    except Exception as _ee:
        logger.debug("vllm.apply_kv_cache_config_defaults skipped: %s", _ee)

    if _bool_opt(options, "perf_count", False):
        config["PERF_COUNT"] = "YES"

    _num_threads = os.environ.get("OV_INFERENCE_NUM_THREADS")
    if device == "CPU" and _num_threads and "INFERENCE_NUM_THREADS" not in config:
        config["INFERENCE_NUM_THREADS"] = int(_num_threads)

    # Dump pre-plugin IR for RoPE pattern analysis
    _dump_dir = os.environ.get("OV_DUMP_PRE_PLUGIN_IR")
    if _dump_dir:
        os.makedirs(_dump_dir, exist_ok=True)
        import hashlib as _hl
        _tag = _hl.sha256(str([list(i.get_partial_shape()) for i in om.inputs]).encode()).hexdigest()[:12]
        _path = os.path.join(_dump_dir, f"pre_plugin_{_tag}")
        if not os.path.isfile(_path + ".xml"):
            serialize(om, _path + ".xml", _path + ".bin")
            print(f"[OV_DUMP] Dumped pre-plugin IR to {_path}.xml", flush=True)

    compiled = core.compile_model(om, device, config)
    # Keep the widened affinity mask — TBB threads were created during
    # compile_model and inherit the mask at creation time. Restoring the
    # narrow mask would not shrink the thread pool but would prevent newly
    # spawned worker threads from running on most cores. Leave it wide.
    logger.debug(f"OpenVINO graph compile successful on device {device}")

    _post_dir = os.environ.get("OV_DUMP_POST_PLUGIN_IR")
    if _post_dir:
        os.makedirs(_post_dir, exist_ok=True)
        import hashlib as _hl2
        _tag2 = _hl2.sha256(str([list(i.get_partial_shape()) for i in om.inputs]).encode()).hexdigest()[:12]
        _path2 = os.path.join(_post_dir, f"post_plugin_{_tag2}")
        if not os.path.isfile(_path2 + ".xml"):
            rm_dump = compiled.get_runtime_model()
            serialize(rm_dump, _path2 + ".xml", _path2 + ".bin")
            print(f"[OV_DUMP] Dumped post-plugin runtime IR to {_path2}.xml", flush=True)

    return compiled
