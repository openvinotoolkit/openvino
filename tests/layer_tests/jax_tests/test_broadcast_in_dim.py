# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestBroadcastInDim(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, target_shape, dimensions):
        self.input_shape = input_shape

        def jax_broadcast_in_dim(inp):
            out = lax.broadcast_in_dim(inp, target_shape, dimensions)
            return out

        return jax_broadcast_in_dim, None, 'broadcast_in_dim'

    test_data_basic = [
        dict(input_shape=[2, 3], target_shape=[1, 2, 3], dimensions=[1, 2]),
        dict(input_shape=[2, 3], target_shape=[2, 3, 1], dimensions=[0, 1]),
        dict(input_shape=[2, 3], target_shape=[2, 1, 3], dimensions=[0, 2]),
        dict(input_shape=[2, 3], target_shape=[2, 1, 1, 3], dimensions=[0, 3]),
        dict(input_shape=[2, 3], target_shape=[2, 2, 3, 3], dimensions=[0, 2]),
        dict(input_shape=[2, 3], target_shape=[2, 2, 3, 3], dimensions=[0, 3]),
        dict(input_shape=[2, 3], target_shape=[2, 2, 3, 3], dimensions=[1, 2]),
        dict(input_shape=[2, 3], target_shape=[2, 2, 3, 3], dimensions=[1, 3]),
        dict(input_shape=[2, 3], target_shape=[2, 3, 2, 3], dimensions=[0, 1]),
        dict(input_shape=[2, 3], target_shape=[2, 3, 2, 3], dimensions=[2, 3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_broadcast_in_dim(self, ie_device, precision, ir_version, params):
        self._test(*self.create_model(**params),
                   ie_device, precision,
                   ir_version)
