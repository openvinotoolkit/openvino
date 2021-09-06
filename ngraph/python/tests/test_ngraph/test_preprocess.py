# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

import ngraph as ng
from ngraph.preprocess import PrePostProcessor, InputInfo, PreProcessSteps
from ngraph import Function, PartialShape
# from ngraph.impl import Function, PartialShape, Shape, Type
#from ngraph.impl.op import Parameter
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


def test_ngraph_preprocess1():
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

    input_data = np.array([[1,2], [3,4]]).astype(np.float32)
    expected_output = np.array([[0,1], [2,3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    print("In: {}".format(output))
    print("Out: {}".format(output))
    assert np.equal(output, expected_output).all()
