# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestRsqrt(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape):
        self.input_shape = input_shape

        def jax_rsqrt(inp):
            out = lax.rsqrt(inp)
            return out

        return jax_rsqrt, None, 'rsqrt'

    @pytest.mark.parametrize("input_shape", [
        [10],
        [2, 3],
        [3, 4, 5],
        [7, 16, 11, 20],
        [6, 5, 4, 3, 2],
        [1, 1, 9, 1]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_rsqrt(self, ie_device, precision, ir_version, input_shape):
        self._test(*self.create_model(input_shape=input_shape),
                   ie_device, precision,
                   ir_version)
