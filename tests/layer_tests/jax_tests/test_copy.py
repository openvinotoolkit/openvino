# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestCopy(JaxLayerTest):
    def _prepare_input(self):
        lhs = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        rhs = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [lhs, rhs]

    def create_model(self, input_shape):
        self.input_shape = input_shape

        def jax_copy(lhs, rhs):
            add = lhs + rhs
            copy = jnp.array(add)
            return lax.rsqrt(copy)

        return jax_copy, None, 'copy'

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
    def test_copy(self, ie_device, precision, ir_version, input_shape):
        self._test(*self.create_model(input_shape=input_shape),
                   ie_device, precision,
                   ir_version)
