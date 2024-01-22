# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import re

import openvino.runtime.opset13 as ov
from openvino import Type


@pytest.mark.parametrize("prefix_string", [
    "",
    "ABC",
    "custom_prefix_",
])
@pytest.mark.parametrize("suffix_string", [
    "",
    "XYZ",
    "_custom_suffix",
])
def test_fake_quantize_affix(prefix_string, suffix_string):
    levels = np.int32(4)
    data_shape = [1, 2, 3, 4]
    bound_shape = []

    data_name = "data"
    parameter_data = ov.parameter(data_shape, name=data_name, dtype=np.float32)

    input_low_name = "input_low"
    parameter_input_low = ov.parameter(bound_shape, name=input_low_name, dtype=np.float32)

    input_high_name = "input_high"
    parameter_input_high = ov.parameter(bound_shape, name=input_high_name, dtype=np.float32)

    output_low_name = "output_low"
    parameter_output_low = ov.parameter(bound_shape, name=output_low_name, dtype=np.float32)

    output_high_name = "output_high"
    parameter_output_high = ov.parameter(bound_shape, name=output_high_name, dtype=np.float32)

    model = ov.fake_quantize(
        parameter_data,
        parameter_input_low,
        parameter_input_high,
        parameter_output_low,
        parameter_output_high,
        levels,
        prefix=prefix_string,
        suffix=suffix_string,
    )
    
    # Check if node was created correctly
    assert model.get_type_name() == "FakeQuantize"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]
    # Check that data parameter and node itself do not change:
    if prefix_string != "":
        assert prefix_string not in model.friendly_name
        assert prefix_string not in parameter_data.friendly_name
    if suffix_string != "":
        assert suffix_string not in model.friendly_name
        assert suffix_string not in parameter_data.friendly_name
    # Check that other parameters change:
    assert prefix_string + input_low_name + suffix_string == parameter_input_low.friendly_name
    assert prefix_string + input_high_name + suffix_string == parameter_input_high.friendly_name
    assert prefix_string + output_low_name + suffix_string == parameter_output_low.friendly_name
    assert prefix_string + output_high_name + suffix_string == parameter_output_high.friendly_name
