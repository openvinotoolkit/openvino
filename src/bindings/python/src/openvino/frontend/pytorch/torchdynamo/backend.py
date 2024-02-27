# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
from functools import partial
from hashlib import sha256

import torch
from torch._dynamo.backends.common import fake_tensor_unsupported, aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import decomposition_table, get_decompositions

from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo import decompositions
from openvino.frontend.pytorch.torchdynamo.decompositions import get_aot_decomposition_list
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, openvino_compile_cached_model
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_model_caching, _get_decompositions, _get_aot_autograd

from openvino.runtime import Core, Type, PartialShape

log = logging.getLogger(__name__)

"""
    This is a preview feature in OpenVINO. This feature
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.torch
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""

openvino_options = {}

@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs, options=None):
    if (_get_aot_autograd(options)):
        global openvino_options
        openvino_options = options
        return aot_autograd(fw_compiler=fx_openvino, bw_compiler=fx_openvino)(subgraph, example_inputs)
    return fx_openvino(subgraph, example_inputs, options)

@register_backend
@fake_tensor_unsupported
def openvino_ts(subgraph, example_inputs):
    return ts_openvino(subgraph, example_inputs)

def ts_openvino(subgraph, example_inputs):
    try:
        model = torch.jit.script(subgraph)
        model.eval()
        fr_model = torch.jit.freeze(model)

        core = Core()
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework('pytorch')
        dtype_mapping = {
            torch.float64: Type.f64,
            torch.float32: Type.f32,
            torch.float16: Type.f16,
            torch.int64: Type.i64,
            torch.int32: Type.i32,
            torch.uint8: Type.u8,
            torch.int8: Type.i8,
            torch.bool: Type.boolean,
        }
        decoder = TorchScriptPythonDecoder(fr_model)

        # TODO: Use convert_model instead when mo --convert_model api becomes a part of OV runtime
        im = fe.load(decoder)
        om = fe.convert(im)

        for idx, input_data in enumerate(example_inputs):
            om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
        om.validate_nodes_and_infer_types()

        device = "CPU"
        if (os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None):
            device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
            assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

        compiled_model = core.compile_model(om, device)

        def _call(*args):
            if not hasattr(_call, "execute_on_ov"):
                _call.execute_on_ov = True
            execute_on_ov = getattr(_call, "execute_on_ov")
            if execute_on_ov:
                ov_inputs = [a.detach().cpu().numpy() for a in args]
                try:
                    res = compiled_model(ov_inputs)
                except Exception as e:
                    log.debug(f"Failed in OpenVINO execution: {e}")
                    _call.execute_on_ov = False
                    return subgraph.forward(*args)
                result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
                return result
            else:
                return subgraph.forward(*args)
        return _call
    except Exception as e:
        log.debug(f"Failed in compilation: {e}")
        return compile_fx(subgraph, example_inputs)

def fx_openvino(subgraph, example_inputs, options=None):
    try:
        if len(openvino_options) != 0:
            options = openvino_options
        executor_parameters = None
        inputs_reversed = False
        openvino_model_caching = _get_model_caching(options)
        if openvino_model_caching is not None and openvino_model_caching:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", _get_device(options), example_inputs, _get_cache_dir(options))
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, options, *example_inputs)
                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res
                return _call
        if inputs_reversed:
            example_inputs.reverse()

        from torch._subclasses.fake_tensor import FakeTensorMode
        decompositions = _get_decompositions(options)
        if (_get_aot_autograd(options)):
            decompositions = decompositions + get_aot_decomposition_list()
        with FakeTensorMode(allow_non_fake_inputs=True):
            model = make_fx(subgraph, decomposition_table=get_decompositions(decompositions))(*example_inputs)

        with torch.no_grad():
            model.eval()
        partitioner = Partitioner(options)
        compiled_model = partitioner.make_partitions(model, options)

        if executor_parameters is not None and 'model_hash_str' in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            res = execute(compiled_model, *args, executor="openvino",
                          executor_parameters=executor_parameters, options=options)
            return res
        return _call
    except Exception as e:
        log.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)

def reset():
    clear_caches()
