# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestReduceWindowMax(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, window_dimensions, window_strides, padding):
        self.input_shape = input_shape

        def jax_reduce_window_max(inp):
            out = lax.reduce_window(
                inp,
                -jnp.inf,
                lax.max,
                window_dimensions=window_dimensions,
                window_strides=window_strides,
                padding=padding
            )
            return out

        return jax_reduce_window_max, None, 'reduce_window_max'

    test_data_basic = [
        dict(input_shape=[1, 112, 112, 64], window_dimensions=[1, 3, 3, 1]),
        dict(input_shape=[1, 112, 112, 64], window_dimensions=[1, 2, 2, 1]),
        dict(input_shape=[1, 60, 85, 16], window_dimensions=[1, 3, 4, 1]),
        dict(input_shape=[1, 100, 50, 16], window_dimensions=[1, 3, 2, 1]),
    ]

    @pytest.mark.parametrize("padding", [
        'SAME', 'VALID'
    ])
    @pytest.mark.parametrize("window_strides", [
        [1, 2, 2, 1], [1, 5, 5, 1], [1, 2, 3, 1], [1, 3, 1, 1]
    ])
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit_jax_fe
    def test_reduce_window_max(self, ie_device, precision, ir_version, params, padding, window_strides):
        self._test(*self.create_model(**params, padding=padding,
                                      window_strides=window_strides),
                   ie_device, precision,
                   ir_version)
