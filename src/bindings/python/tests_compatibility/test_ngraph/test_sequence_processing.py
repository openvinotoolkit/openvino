# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type


def test_onehot():
    param = ng.parameter([3], dtype=np.int32)
    # output type is derived from 'on_value' and 'off_value' element types
    # Need to set explicitly 'on_value' and 'off_value' types.
    # If we don't do it explicitly, depending on OS/packages versions types can be unpredictably either int32 or int64
    on_value = np.array(1, dtype=np.int64)
    off_value = np.array(0, dtype=np.int64)
    depth = 3
    axis = 0
    model = ng.one_hot(param, depth, on_value, off_value, axis)
    assert model.get_output_size() == 1
    assert model.get_type_name() == "OneHot"
    assert list(model.get_output_shape(0)) == [3, 3]
    assert model.get_output_element_type(0) == Type.i64
