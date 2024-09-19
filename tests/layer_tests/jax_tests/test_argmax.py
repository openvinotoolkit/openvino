# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


def generate_shape_axis_pairs(input_shapes):
    test_data = []
    for shape in input_shapes:
        rank = len(shape)
        # Only [0, rank - 1] are valid axes for lax.argmax
        valid_axes = range(0, rank)
        test_data.extend([{'input_shape': shape, 'axis': axis}
                         for axis in valid_axes])
    return test_data


class TestArgmax(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, axis, index_dtype):
        self.input_shape = input_shape

        def jax_argmax(inp):
            out = lax.argmax(inp, axis, index_dtype)
            return out

        return jax_argmax, None, 'argmax'

    input_shapes = [
        [64],
        [64, 16],
        [48, 23, 54],
        [2, 18, 32, 25],
        [2, 18, 32, 25, 128],
    ]
    test_data = generate_shape_axis_pairs(input_shapes)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_argmax(self, ie_device, precision, ir_version, params, index_dtype):
        self._test(*self.create_model(**params, index_dtype=index_dtype),
                   ie_device, precision,
                   ir_version)
