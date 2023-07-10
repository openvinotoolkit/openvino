# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
import torch
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx

from openvino.frontend import FrontEndManager
from openvino.runtime import Core, Type, PartialShape
from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

log = logging.getLogger(__name__)

"""
    This is a preview feature in OpenVINO. Torchscript backend
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.frontend.pytorch.torchdynamo.backend
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""


@register_backend
@fake_tensor_unsupported
def openvino(subgraph, example_inputs):
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
        if (os.getenv("OPENVINO_TS_BACKEND_DEVICE") is not None):
            device = os.getenv("OPENVINO_TS_BACKEND_DEVICE")
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
