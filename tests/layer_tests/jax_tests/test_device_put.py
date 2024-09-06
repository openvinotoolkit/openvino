# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(233454)


class TestDevicePut(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            x = rng.uniform(-5.0, 5.0, self.input_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            x = rng.integers(-8, 8, self.input_shape).astype(self.input_type)
        else:
            x = rng.integers(0, 8, self.input_shape).astype(self.input_type)

        x = jnp.array(x)
        return [x]

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_device_put(x):
            x = x + 1
            return jax.device_put(x)

        return jax_device_put, None, 'device_put'

    @pytest.mark.parametrize("input_shape", [[2], [3, 4]])
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_device_put(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)
