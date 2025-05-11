# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_tensor_value_info
from openvino.frontend import FrontEndManager


def make_onnx_model(model_name: str) -> None:
    """Make onnyx model file as `model_name`."""
    # Input model
    add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["out1"], name="onnx_add_op")

    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
    ]
    graph = make_graph([add], "test_graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend", opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx.save_model(model, model_name)


@pytest.fixture(scope="module", params=["onnx"])
def frontend_model(request):
    """Fixture return frontend name and test model parameters.

    Model parameters:
       - model name

    If frontend name no in the supported frontends then tests will be skipped.
    """
    frontend = request.param
    if frontend not in FrontEndManager().get_available_front_ends():
        pytest.skip(allow_module_level=True)
        return frontend, ""

    models = {"onnx": ("input_model", make_onnx_model)}
    model_name, make_model = models.get(frontend)
    make_model(model_name)
    yield frontend, model_name
    os.remove(model_name)


def test_delete_place_as_last(frontend_model):
    """Place object must be deleted as last to check if it keep dependency on frontend library.

    Verify issue CVS-82282.
    """
    frontend, model = frontend_model

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework=frontend)
    model = fe.load(model)
    place = model.get_place_by_tensor_name(tensor_name="out1")

    del fem
    del fe
    del model
    assert place


def test_delete_model_as_last(frontend_model):
    """Model object must be deleted as last to check if it keep dependency on frontend library.

    Verify issue CVS-82282.
    """
    frontend, model = frontend_model

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework=frontend)
    model = fe.load(model)

    del fe
    del fem
    assert model
