# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import json

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Function, PartialShape, Shape
from ngraph.impl.passes import Manager
from tests_compatibility.test_ngraph.util import count_ops_of_type


def test_constant_folding():
    node_constant = ng.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ng.ceiling(node_constant)
    func = Function(node_ceil, [], "TestFunction")

    assert count_ops_of_type(func, node_ceil) == 1
    assert count_ops_of_type(func, node_constant) == 1

    pass_manager = Manager()
    pass_manager.register_pass("ConstantFolding")
    pass_manager.run_passes(func)

    assert count_ops_of_type(func, node_ceil) == 0
    assert count_ops_of_type(func, node_constant) == 1

    new_const = func.get_results()[0].input(0).get_source_output().get_node()

    values_out = new_const.get_vector()
    values_expected = [0.0, 1.0, 0.0, -2.0, 3.0, 3.0]

    assert np.allclose(values_out, values_expected)
