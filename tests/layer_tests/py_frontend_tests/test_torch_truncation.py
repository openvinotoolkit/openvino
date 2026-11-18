# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import openvino as ov
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder


@pytest.mark.parametrize("mode", ["script", "script_untyped", "export"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("operation", ["div", "trunc", "trunc_"])
def test_truncation_integer_graph(mode, dtype, operation):
    class Model(torch.nn.Module):
        __constants__ = ["operation"]

        def __init__(self):
            super().__init__()
            self.operation = operation

        def forward(self, x, y):
            if self.operation == "div":
                return torch.div(x, y, rounding_mode="trunc")
            if self.operation == "trunc_":
                return torch.trunc_(x.clone())
            return torch.trunc(x)

    x = torch.tensor([-7, 7, -9, 9, -1], dtype=dtype)
    y = torch.tensor([2, -2, 4, 4, 3], dtype=dtype)
    if dtype == torch.int64:
        x[0] = 5000000001
    if dtype == torch.float32:
        x = torch.tensor([1e10, -1e10, float("inf"), float("nan"), -7.5])
    model = Model()
    frontend = FrontEndManager().load_by_framework("pytorch")
    if mode != "export":
        source = TorchScriptPythonDecoder(torch.jit.script(model))
        input_model = frontend.load(source)
        ov_type = ov.Type.f32 if dtype == torch.float32 else ov.Type.i64 if dtype == torch.int64 else ov.Type.i32
        if mode == "script":
            for place in input_model.get_inputs():
                input_model.set_element_type(place, ov_type)
                input_model.set_partial_shape(place, ov.PartialShape([-1]))
        converted = frontend.convert(input_model)
        if mode == "script_untyped":
            for parameter in converted.get_parameters():
                parameter.set_element_type(ov_type)
                parameter.set_partial_shape(ov.PartialShape([-1]))
            converted.validate_nodes_and_infer_types()
            frontend.normalize(converted)
    else:
        converted = ov.convert_model(torch.export.export(model, (x, y)))

    types = [node.get_type_name() for node in converted.get_ops()]
    for name in ("Less", "Ceiling", "Floor", "Select"):
        assert types.count(name) == (1 if dtype == torch.float32 or mode == "script_untyped" else 0)
    if dtype != torch.float32 and mode != "script_untyped":
        assert "Convert" not in types
    if operation == "div":
        divide = next(node for node in converted.get_ops() if node.get_type_name() == "Divide")
        assert not divide.get_attributes()["m_pythondiv"]
    inputs = [ov.Tensor(x.numpy())]
    if len(converted.inputs) == 2:
        inputs.append(ov.Tensor(y.numpy()))
    output = ov.Tensor(converted.output().get_element_type(), list(x.shape))
    assert converted.evaluate([output], inputs)
    np.testing.assert_array_equal(output.data, model(x.clone(), y).numpy())


@pytest.mark.parametrize("mode", ["script", "export"])
@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_sort_indices_div_trunc(mode, top_k):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = top_k

        def forward(self, top_k_experts):
            _, index_sorted_experts = top_k_experts.sort(0)
            return index_sorted_experts.div(self.top_k, rounding_mode="trunc")

    x = torch.tensor([3, 0, 2, 1, 4, 2], dtype=torch.int64)
    model = Model()
    if mode == "script":
        frontend = FrontEndManager().load_by_framework("pytorch")
        source = TorchScriptPythonDecoder(torch.jit.script(model))
        converted = frontend.convert(frontend.load(source))
        converted.get_parameters()[0].set_element_type(ov.Type.i64)
        converted.get_parameters()[0].set_partial_shape(ov.PartialShape([-1]))
        converted.validate_nodes_and_infer_types()
    else:
        converted = ov.convert_model(torch.export.export(model, (x,)))
    types = [node.get_type_name() for node in converted.get_ops()]
    assert types.count("TopK") == 1
    assert types.count("Divide") == 1
    assert all(name not in types for name in ("Less", "Ceiling", "Floor", "Select"))
    assert converted.output().get_element_type() == ov.Type.i64
    output = ov.Tensor(ov.Type.i64, list(x.shape))
    assert converted.evaluate([output], [ov.Tensor(x.numpy())])
    np.testing.assert_array_equal(output.data, model(x).numpy())
