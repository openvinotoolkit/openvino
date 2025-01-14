# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestSlice(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, start_indices, limit_indices):
        self.input_shape = input_shape

        def jax_slice(inp):
            out = lax.slice(inp, start_indices, limit_indices)
            return out

        return jax_slice, None, 'slice'

    @pytest.mark.parametrize("input_shape", [
        [16, 16, 16, 16],
    ])
    @pytest.mark.parametrize("start_indices", [
        [0, 0, 0, 0], [0, 1, 0, 2], [2, 2, 1, 0], [2, 3, 5, 6],
        [5, 0, 4, 2]
    ])
    @pytest.mark.parametrize("limit_indices", [
        [8, 8, 8, 8], [8, 9, 10, 11], [13, 13, 12, 12], [15, 15, 14, 15]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_slice(self, ie_device, precision, ir_version, input_shape, start_indices, limit_indices):
        self._test(
            *self.create_model(input_shape=input_shape, start_indices=start_indices, limit_indices=limit_indices),
            ie_device, precision,
            ir_version)


class TestSliceWithStrides(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, start_indices, limit_indices, strides):
        self.input_shape = input_shape

        def jax_slice_with_strides(inp):
            out = lax.slice(inp, start_indices, limit_indices, strides)
            return out

        return jax_slice_with_strides, None, 'slice'

    @pytest.mark.parametrize("input_shape", [
        [16, 16, 16, 16],
    ])
    @pytest.mark.parametrize("start_indices", [
        [0, 0, 0, 0], [0, 1, 0, 2], [2, 2, 1, 0], [2, 3, 5, 6],
        [5, 0, 4, 2]
    ])
    @pytest.mark.parametrize("limit_indices", [
        [8, 8, 8, 8], [8, 9, 10, 11], [13, 13, 12, 12], [15, 15, 14, 15]
    ])
    @pytest.mark.parametrize("strides", [
        [1, 1, 1, 1], [1, 2, 3, 4], [10, 15, 12, 3], [8, 5, 4, 1]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_slice_with_strides(self, ie_device, precision, ir_version, input_shape, start_indices, limit_indices,
                                strides):
        self._test(*self.create_model(input_shape=input_shape, start_indices=start_indices,
                                      limit_indices=limit_indices, strides=strides),
                   ie_device, precision,
                   ir_version)
