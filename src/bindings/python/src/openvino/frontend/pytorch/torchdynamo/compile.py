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
    # than using the unset Parameter upper-bound of 0 (which collapses sizes
    # like [arg99_1, 2048] to [0, 2048] at runtime).
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

    compiled = core.compile_model(om, device, config)
    logger.debug(f"OpenVINO graph compile successful on device {device}")

    return compiled
