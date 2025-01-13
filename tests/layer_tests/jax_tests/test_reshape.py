# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestReshape(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, new_sizes):
        self.input_shape = input_shape

        def jax_reshape(inp):
            out = lax.reshape(inp, new_sizes=new_sizes)
            return out

        expected_op = 'reshape'
        if input_shape == new_sizes:
            expected_op = None

        return jax_reshape, None, expected_op

    @pytest.mark.parametrize("input_shape", [
        [64],
        [4, 16],
        [4, 4, 4],
        [2, 8, 2, 2]
    ])
    @pytest.mark.parametrize("new_sizes", [
        [64], [64, 1], [16, 4], [8, 1, 8], [1, 1, 64], [8, 2, 2, 2], [2, 4, 2, 2, 2]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_reshape(self, ie_device, precision, ir_version, input_shape, new_sizes):
        self._test(*self.create_model(input_shape=input_shape, new_sizes=new_sizes),
                   ie_device, precision,
                   ir_version)


class TestReshapeWithDimensions(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, new_sizes, dimensions):
        self.input_shape = input_shape

        def jax_reshape_with_dimensions(inp):
            out = lax.reshape(inp, new_sizes=new_sizes, dimensions=dimensions)
            return out

        expected_op = 'reshape'
        if input_shape == new_sizes:
            expected_op = None

        return jax_reshape_with_dimensions, None, expected_op

    @pytest.mark.parametrize("input_shape", [
        [4, 16, 1],
        [8, 4, 2],
        [2, 1, 32],
        [8, 1, 8],
        [16, 2, 2]
    ])
    @pytest.mark.parametrize("new_sizes", [
        [64], [64, 1], [16, 4], [8, 1, 8], [1, 1, 64], [8, 2, 2, 2], [2, 4, 2, 2, 2]
    ])
    @pytest.mark.parametrize("dimensions", [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_reshape_with_dimensions(self, ie_device, precision, ir_version, input_shape, new_sizes, dimensions):
        self._test(*self.create_model(input_shape=input_shape, new_sizes=new_sizes, dimensions=dimensions),
                   ie_device, precision,
                   ir_version)
