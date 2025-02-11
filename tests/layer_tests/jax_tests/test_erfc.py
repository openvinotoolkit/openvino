# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(109734)


class TestErfc(JaxLayerTest):
    def _prepare_input(self):
        # erf are mostly changing in a range [-4, 4]
        x = rng.uniform(-4.0, 4.0, self.input_shape).astype(self.input_type)

        x = jnp.array(x)
        return [x]

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_erfc(x):
            return jax.lax.erfc(x)

        return jax_erfc, None, 'erfc'

    @pytest.mark.parametrize("input_shape", [[2], [3, 4]])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit_jax_fe
    def test_erfc(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)
