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
    # vLLM's init_cpu_threads_env pins the worker process to a single CPU
    # via sched_setaffinity, which TBB/OV sample on first parallel use →
    # INFERENCE_NUM_THREADS=1 regardless of what we request. Widen the mask
    # once, before our first Core() compile so TBB sizes its thread pool
    # to the full available set. Default off because widening races with
    # vLLM's dummy-run shape inference for PA; enable via env when the
    # full warm-up path has been tested for this model.
    if os.environ.get("OV_UNBIND_AFFINITY", "0") == "1":
        try:
            os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
            if os.environ.get("OV_DBG_CONFIG"):
                print(f"[AFFINITY] widened to {os.cpu_count()} cores", flush=True)
        except Exception as _ea:
            if os.environ.get("OV_DBG_CONFIG"):
                print(f"[AFFINITY] failed: {_ea}", flush=True)
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

        # Register any dangling Parameters created by the paged_attention
        # translator. These carry side-channel inputs (KV cache / block tables
        # / past_lens / etc.) that will be bound at infer time from vLLM's
        # ForwardContext. The translator tags them with friendly-name prefix
        # "__pa__". Without this registration the Model fails validation with
        # "unregistered_parameters" errors.
        try:
            _existing_ids = {id(p) for p in om.get_parameters()}
            _to_add = []
            for _node in om.get_ordered_ops():
                if _node.get_type_name() != "Parameter":
                    continue
                if id(_node) in _existing_ids:
                    continue
                _fn = _node.get_friendly_name()
                if _fn.startswith("__pa__"):
                    _to_add.append(_node)
            import os as _os_pa
            if _os_pa.environ.get("OV_DBG_PA_PARAMS"):
                print(f"[PA_PARAMS] model has {len(om.get_parameters())} existing params, adding {len(_to_add)} PA params", flush=True)
                for _p in _to_add[:5]:
                    print(f"  PA param: {_p.get_friendly_name()}", flush=True)
            if _to_add:
                om.add_parameters(_to_add)
        except Exception as _ee:
            logger.debug(f"PA parameter registration skipped: {_ee}")

        # Some FX graphs (notably vLLM's symint-heavy ones) emit Unsqueeze
        # wrappers that leave rank-mismatched Concat inputs for list-construct
        # nodes. Strip redundant Unsqueezes whose inner input is already rank>=1
        # so that validate_nodes_and_infer_types() succeeds.
        def _rank_ge_1(_val):
            _n = _val.get_node()
            _ps = _val.get_partial_shape()
            if _ps.rank.is_static and _ps.rank.get_length() >= 1:
                return True
            if _n.get_type_name() == "Constant":
                return len(_n.get_output_shape(0)) >= 1
            return False

        try:
            for _ in range(64):
                try:
                    om.validate_nodes_and_infer_types()
                    break
                except Exception:
                    pass
                _made_change = False
                for _node in list(om.get_ordered_ops()):
                    if _node.get_type_name() != "Concat":
                        continue
                    if _node.get_input_size() < 2:
                        continue
                    for _i in range(_node.get_input_size()):
                        _src = _node.input_value(_i)
                        _src_node = _src.get_node()
                        if _src_node.get_type_name() != "Unsqueeze":
                            continue
                        _inner = _src_node.input_value(0)
                        if _rank_ge_1(_inner):
                            _node.input(_i).replace_source_output(_inner)
                            _made_change = True
                if not _made_change:
                    break
        except Exception as _ee:
            logger.debug(f"concat-rank normalization skipped: {_ee}")

        # Rewrite MatMul(X, Transpose(Const_fp16)) to the oneDNN-BRGEMM-friendly
        # form: MatMul(X, Convert(Const_fp16 -> f32)) with transpose_b=true.
        # Without this, CPU plugin falls back to gemm_mlas_f32 (6x slower than
        # brgemm_avx512_f32) for fp16-activation FCs. OV GenAI's IR has exactly
        # this pattern; we emulate it.
        if os.environ.get("OV_FC_DECOMPRESS", "1") == "1":
            try:
                from openvino import opset1 as _o1
                import numpy as _np
                _rewritten = 0
                for _mm in list(om.get_ordered_ops()):
                    if _mm.get_type_name() != "MatMul":
                        continue
                    try:
                        _tb = _mm.get_transpose_b()
                        _ta = _mm.get_transpose_a()
                    except Exception:
                        continue
                    if _tb:
                        continue  # already transpose_b=true
                    _src = _mm.input_value(1).get_node()
                    _const = None
                    _new_tb = False
                    if _src.get_type_name() == "Transpose":
                        _inner = _src.input_value(0).get_node()
                        if _inner.get_type_name() == "Constant":
                            _perm_node = _src.input_value(1).get_node()
                            if _perm_node.get_type_name() == "Constant":
                                _perm = list(_perm_node.get_data().flatten())
                                if _perm == [1, 0]:
                                    _const = _inner
                                    _new_tb = True
                    elif _src.get_type_name() == "Constant":
                        _const = _src
                    if _const is None:
                        continue
                    if _const.get_element_type() != Type.f16:
                        continue
                    # Keep weight as FP16. With INFERENCE_PRECISION_HINT=f16,
                    # the plugin folds/collapses alternative weight dtypes
                    # back to fp16 regardless. BF16 hint would unlock GenAI's
                    # brgemm_avx512_bf16-style dispatch, but our PA pipeline
                    # rejects bf16 in several node validators so we can't
                    # enable it globally.
                    _new_const = _const
                    _conv_w = _o1.convert(_new_const.output(0), "f32")
                    # Mark Convert as decompression so CPU plugin's
                    # ConvertMatMulToFC pass accepts this pattern and routes
                    # to brgemm_avx512_f32 (fast) instead of gemm_mlas_f32.
                    try:
                        # RTMap is keyed by DiscreteTypeInfo::operator std::string()
                        # which is "<name>_<version_id>". For Decompression this is
                        # "decompression_0" — matching the key is_decompression()
                        # uses internally.
                        _conv_w.get_rt_info()["decompression_0"] = True
                    except Exception:
                        pass
                    _mm.input(1).replace_source_output(_conv_w.output(0))
                    # Upcast activation to f32 just for this FC; cast the
                    # output back to fp16 so downstream ops still see fp16.
                    # (Tried rank-3 Unsqueeze[1,N,K] to match GenAI's IR, but
                    # CPU plugin's AlignMatMulInputRanks + ConvertMatMulToFC
                    # normalize back to rank-2 regardless, so no-op.)
                    _act_src = _mm.input_value(0)
                    if _act_src.get_element_type() == Type.f16:
                        _conv_a = _o1.convert(_act_src, "f32")
                        _mm.input(0).replace_source_output(_conv_a.output(0))
                        _out = _mm.output(0)
                        _consumers = list(_out.get_target_inputs())
                        _down = _o1.convert(_out, "f16")
                        for _cin in _consumers:
                            _cin.replace_source_output(_down.output(0))
                    try:
                        _mm.set_transpose_b(_new_tb if _new_tb else _mm.get_transpose_b())
                    except Exception:
                        pass
                    _rewritten += 1
                if os.environ.get("OV_DBG_FC"):
                    print(f"[FC_DECOMPRESS] rewrote {_rewritten} MatMuls", flush=True)
                om.validate_nodes_and_infer_types()
            except Exception as _ee:
                if os.environ.get("OV_DBG_FC"):
                    import traceback as _tb
                    print(f"[FC_DECOMPRESS] error: {_ee}", flush=True)
                    _tb.print_exc()

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
    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            continue  # Already baked as Constant above
        # Use the concrete runtime tensor shape so OV can propagate exact
        # shape bounds through the graph.
        om.inputs[_tensor_idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[_tensor_idx].get_node().set_partial_shape(PartialShape(list(input_data.size())))
        _tensor_idx += 1

    om.validate_nodes_and_infer_types()

    config = _get_config(options)

    if model_hash_str is not None:
        if not _is_cache_dir_in_config(options):
            config["CACHE_DIR"] = cache_root

    # KV cache at f32 (verified-correct for PA). Hardware has no bf16 support,
    # so matmuls run f32 by default — use DYNAMIC_QUANTIZATION_GROUP_SIZE=32
    # to quantize FC activations to int8 on the fly (vnni int8 GEMM is much
    # faster than f32 GEMM). Matches OV GenAI's CPU speedup mechanism.
    if device == "CPU" and "KV_CACHE_PRECISION" not in config:
        config["KV_CACHE_PRECISION"] = os.environ.get("OV_KV_CACHE_PRECISION", "f32")
    if device == "CPU" and "DYNAMIC_QUANTIZATION_GROUP_SIZE" not in config:
        config["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = int(
            os.environ.get("DYNAMIC_QUANTIZATION_GROUP_SIZE", "32"))
    # Let plugin pick oneDNN fp16 GEMM for FCs (model weights arrive as fp16
    # from vLLM). PA op is fenced with Convert(f32) in the translator, so it
    # stays f32 regardless of this hint. Overridable via env.
    _inf_hint = os.environ.get("OV_INFERENCE_PRECISION_HINT", "f16")
    if device == "CPU" and "INFERENCE_PRECISION_HINT" not in config and _inf_hint:
        config["INFERENCE_PRECISION_HINT"] = _inf_hint

    if os.environ.get("OV_PERF_COUNT"):
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

    if os.environ.get("OV_DBG_CONFIG"):
        print(f"[CONFIG] {config}", flush=True)
    compiled = core.compile_model(om, device, config)
    if os.environ.get("OV_DBG_CONFIG"):
        try:
            print(f"[CONFIG] actual INFERENCE_NUM_THREADS={compiled.get_property('INFERENCE_NUM_THREADS')}", flush=True)
            print(f"[CONFIG] actual NUM_STREAMS={compiled.get_property('NUM_STREAMS')}", flush=True)
        except Exception as _e:
            print(f"[CONFIG] get_property failed: {_e}", flush=True)
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
