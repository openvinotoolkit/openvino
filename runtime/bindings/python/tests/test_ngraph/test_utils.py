# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import ngraph as ng
from ngraph.impl import Shape


def test_get_constant_from_source_success():
    dtype = np.int
    input1 = ng.parameter(Shape([5, 5]), dtype=dtype, name="input_1")
    input2 = ng.parameter(Shape([25]), dtype=dtype, name="input_2")
    shape_of = ng.shape_of(input2, name="shape_of")
    reshape = ng.reshape(input1, shape_of, special_zero=True)
    folded_const = ng.impl.util.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is not None
    assert folded_const.get_vector() == [25]


def test_get_constant_from_source_failed():
    dtype = np.int
    input1 = ng.parameter(Shape([5, 5]), dtype=dtype, name="input_1")
    input2 = ng.parameter(Shape([1]), dtype=dtype, name="input_2")
    reshape = ng.reshape(input1, input2, special_zero=True)
    folded_const = ng.impl.util.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is None
