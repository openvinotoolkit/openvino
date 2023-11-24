# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime as ov
import pytest
from openvino._pyopenvino.util import deprecation_warning
from openvino import Shape


def test_get_constant_from_source_success():
    input1 = ov.opset8.parameter(Shape([5, 5]), dtype=int, name="input_1")
    input2 = ov.opset8.parameter(Shape([25]), dtype=int, name="input_2")
    shape_of = ov.opset8.shape_of(input2, name="shape_of")
    reshape = ov.opset8.reshape(input1, shape_of, special_zero=True)
    folded_const = ov.utils.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is not None
    assert folded_const.get_vector() == [25]


def test_get_constant_from_source_failed():
    input1 = ov.opset8.parameter(Shape([5, 5]), dtype=int, name="input_1")
    input2 = ov.opset8.parameter(Shape([1]), dtype=int, name="input_2")
    reshape = ov.opset8.reshape(input1, input2, special_zero=True)
    folded_const = ov.utils.get_constant_from_source(reshape.input(1).get_source_output())

    assert folded_const is None


def test_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="function1 is deprecated"):
        deprecation_warning("function1")
    with pytest.warns(DeprecationWarning, match="function2 is deprecated and will be removed in version 2025.4"):
        deprecation_warning("function2", "2025.4")
    with pytest.warns(DeprecationWarning, match="function3 is deprecated. Use another function instead"):
        deprecation_warning("function3", message="Use another function instead")
    with pytest.warns(DeprecationWarning, match="function4 is deprecated and will be removed in version 2025.4. Use another function instead"):
        deprecation_warning("function4", version="2025.4", message="Use another function instead")
