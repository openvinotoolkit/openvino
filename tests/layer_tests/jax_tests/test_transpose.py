# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestTranspose(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, permutation):
        self.input_shape = input_shape

        def jax_transpose(inp):
            out = lax.transpose(inp, permutation)
            return out

        expected_op = 'transpose'
        if permutation == [0, 1, 2, 3]:
            expected_op = None

        return jax_transpose, None, expected_op

    @pytest.mark.parametrize("input_shape", [
        [7, 16, 11, 20],
        [1, 1, 1, 1],
    ])
    @pytest.mark.parametrize("permutation", [
        [0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
        [1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
        [2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
        [3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_transpose(self, ie_device, precision, ir_version, input_shape, permutation):
        self._test(*self.create_model(input_shape=input_shape, permutation=permutation),
                   ie_device, precision,
                   ir_version)
