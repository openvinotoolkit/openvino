# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl.preprocess import PrePostProcessor, InputInfo, PreProcessSteps, InputTensorInfo
from ngraph import Function, Node, Type
from tests.runtime import get_runtime


def test_ngraph_preprocess_mean():
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")

    function = PrePostProcessor()\
        .input(InputInfo()
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           )
               )\
        .build(function)

    input_data = np.array([[1, 2], [3, 4]]).astype(np.float32)
    expected_output = np.array([[0, 1], [2, 3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_mean_scale_convert():
    shape = [2, 2]
    param1 = ng.parameter(shape, dtype=np.float32, name="A")
    param2 = ng.parameter(shape, dtype=np.float32, name="B")
    function = Function([param1, param2], [param1, param2], "TestFunction")

    def custom_preprocess(node: Node):
        return ng.abs(node)

    function = PrePostProcessor() \
        .input(InputInfo(1)
               .tensor(InputTensorInfo()
                       .set_element_type(Type.i32))
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean(1.)
                           .scale(2.)
                           )
               ) \
        .input(InputInfo(0)
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .custom(custom_preprocess)
                           )
               ) \
        .build(function)

    input_data1 = np.array([[0, 1], [2, -2]]).astype(np.int32)
    input_data2 = np.array([[1, 3], [5, 7]]).astype(np.int32)
    expected_output1 = np.array([[0, 2], [4, 6]]).astype(np.float32)
    expected_output2 = np.array([[-0.5, 0], [0.5, -1.5]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    [output1, output2] = computation(input_data1, input_data2)

    assert np.equal(output1, expected_output1).all()
    assert np.equal(output2, expected_output2).all()


def test_ngraph_invalid_element_type():
    shape = [2, 2]
    param1 = ng.parameter(shape, dtype=np.float32, name="A")
    function = Function([param1], [param1], "TestFunction")
    with pytest.raises(Exception):
        function = PrePostProcessor() \
            .input(InputInfo()
                   .preprocess(PreProcessSteps()
                               .convert_element_type(Type.i32)
                               )
                   ) \
            .build(function)
