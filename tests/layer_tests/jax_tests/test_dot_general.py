# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax

from jax_layer_test_class import JaxLayerTest


class TestDotGeneral(JaxLayerTest):
    def _prepare_input(self):
        lhs = np.random.randint(-50, 50, *self.lhs_shape).astype(self.input_type)
        rhs = np.random.randint(-50, 50, *self.rhs_shape).astype(self.input_type)
        return (lhs, rhs)

    def create_model(self, lhs_shape, rhs_shape, input_type, dimension_numbers):
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.input_type = input_type

        def jax_dot_general(lhs, rhs):
            out = lax.dot_general(lhs, rhs, dimension_numbers)
            return out

        return jax_dot_general, None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("lhs_shape", [[3, 2, 3, 4]])
    @pytest.mark.parametrize("rhs_shape", [[3, 2, 2, 4]])
    @pytest.mark.parametrize("input_type", [np.float32])
    @pytest.mark.parametrize("dimension_numbers", [(((3), (3)), ((0, 1), (0, 1)))])
    def test_dot_general(self, ie_device, precision, ir_version, lhs_shape, rhs_shape, input_type, dimension_numbers):
        self._test(*self.create_model(lhs_shape, rhs_shape, input_type, dimension_numbers), ie_device, precision,
                   ir_version)
