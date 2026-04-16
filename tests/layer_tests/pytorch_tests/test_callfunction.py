# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from pytorch_layer_test_class import PytorchLayerTest


_CALLFUNCTION_CU = torch.jit.CompilationUnit(
    """
def callfunction_const() -> Tensor:
    return torch.ones((2, 3, 4, 5))

def callfunction_relu(x: Tensor) -> Tensor:
    return torch.relu(x)

def callfunction_add(x: Tensor, y: Tensor) -> Tensor:
    return x + y
"""
)


class CallFunctionReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = _CALLFUNCTION_CU.callfunction_relu

    def forward(self, x):
        return self.fn(x)


class CallFunctionAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = _CALLFUNCTION_CU.callfunction_add

    def forward(self, x, y):
        return self.fn(x, y)


class CallFunctionNoArgModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = _CALLFUNCTION_CU.callfunction_const

    def forward(self):
        return self.fn()


class TestCallFunction(PytorchLayerTest):
    def _prepare_input(self, num_inputs=1):
        if num_inputs == 0:
            return ()
        x = self.random.randn(2, 3, 4, 5).astype(np.float32)
        if num_inputs == 1:
            return (x,)
        y = self.random.randn(2, 3, 4, 5).astype(np.float32)
        return (x, y)

    def convert_directly_via_frontend(self, model, example_input, trace_model, dynamic_shapes, ov_inputs, freeze_model):
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        with torch.no_grad():
            if trace_model:
                scripted_model = torch.jit.trace(model, example_input, strict=False)
            else:
                scripted_model = torch.jit.script(model)

        assert self._check_kind_exist(scripted_model.graph, "prim::CallFunction"), (
            f"prim::CallFunction is expected in traced graph, but not found.\nGraph:\n{scripted_model.graph}"
        )

        decoder = TorchScriptPythonDecoder(
            scripted_model,
            graph_element=scripted_model.graph,
            alias_db=scripted_model.graph.alias_db(),
        )
        im = fe.load(decoder)
        om = fe.convert(im)
        self._resolve_input_shape_dtype(om, ov_inputs, dynamic_shapes)
        return scripted_model, om

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_callfunction_no_arg(self, ie_device, precision, ir_version):
        self._test(
            CallFunctionNoArgModel(),
            None,
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            kwargs_to_prepare_input={"num_inputs": 0},
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_callfunction_relu(self, ie_device, precision, ir_version):
        self._test(CallFunctionReluModel(), None, ie_device, precision, ir_version, trace_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_callfunction_add(self, ie_device, precision, ir_version):
        self._test(
            CallFunctionAddModel(),
            None,
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            kwargs_to_prepare_input={"num_inputs": 2},
        )
