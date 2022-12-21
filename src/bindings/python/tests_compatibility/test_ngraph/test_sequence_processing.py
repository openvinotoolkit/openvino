# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type


def test_onehot():
    param = ng.parameter([3], dtype=np.int32)
    model = ng.one_hot(param, 3, 1, 0, 0)
    assert model.get_output_size() == 1
    assert model.get_type_name() == "OneHot"
    assert list(model.get_output_shape(0)) == [3, 3]
    assert model.get_output_element_type(0) == Type.i64
