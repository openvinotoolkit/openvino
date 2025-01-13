# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax

from jax_layer_test_class import JaxLayerTest


class TestDotGeneral(JaxLayerTest):
    def _prepare_input(self):
        lhs = np.random.randint(-10, 10, self.lhs_shape).astype(self.input_type)
        rhs = np.random.randint(-10, 10, self.rhs_shape).astype(self.input_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, dimension_numbers, input_type):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.input_type = input_type

        def jax_dot_general(lhs, rhs):
            out = lax.dot_general(lhs, rhs, dimension_numbers)
            return out

        return jax_dot_general, None, 'dot_general'

    test_data = [
        # 1D vector dot 1D vector
        dict(lhs_shape=[4], rhs_shape=[4], dimension_numbers=(((0), (0)), ((), ()))),
        # matrix mxk dot vector k
        dict(lhs_shape=[2, 5], rhs_shape=[5], dimension_numbers=(((1), (0)), ((), ()))),
        # matrix mxk dot matrix kxn
        dict(lhs_shape=[2, 5], rhs_shape=[5, 6], dimension_numbers=(((1), (0)), ((), ()))),
        # batch matmul case
        dict(lhs_shape=[3, 2, 3, 4], rhs_shape=[3, 2, 2, 4], dimension_numbers=(((3), (3)), ((0, 1), (0, 1)))),
        # batch matmul case: different batch and contracting dimensions
        dict(lhs_shape=[2, 3, 4, 5], rhs_shape=[4, 2, 5, 3], dimension_numbers=(((2, 3), (0, 2)), ((0, 1), (1, 3)))),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    def test_dot_general(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, input_type=input_type), ie_device, precision,
                   ir_version)
