# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(233454)


class TestStopGradient(JaxLayerTest):
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

        def jax_stop_gradient(x):
            x = x + 1
            x = jax.lax.stop_gradient(x)
            return x / 3

        return jax_stop_gradient, None, 'stop_gradient'

    @pytest.mark.parametrize("input_shape", [[2], [3, 4]])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_jax_stop_gradient(self, ie_device, precision, ir_version, input_shape, input_type):
        kwargs = {}
        if input_type == np.float16:
            kwargs["custom_eps"] = 1e-3
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version, **kwargs)
