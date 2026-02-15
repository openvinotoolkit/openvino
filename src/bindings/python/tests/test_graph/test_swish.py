# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import openvino.opset8 as ov
from openvino import Shape, Type


@pytest.mark.parametrize(("beta"), [
    [],
    [ov.parameter(Shape([]), dtype=np.float32, name="beta")]])
def test_swish(beta):
    data = ov.parameter(Shape([3, 10]), dtype=np.float32, name="data")

    node = ov.swish(data, *beta)
    assert node.get_type_name() == "Swish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32
