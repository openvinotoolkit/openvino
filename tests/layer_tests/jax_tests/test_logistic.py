# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(5402)


class TestLogistic(JaxLayerTest):
    def _prepare_input(self):
        
        input = jnp.array(np.random.uniform(-1000, 1000, self.input_shape).astype(self.input_type))
        return [input]

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_logistic(input):
            return jax.lax.logistic(input)

        return jax_logistic, None, 'logistic'

    @pytest.mark.parametrize("input_shape", [[2], [3, 4], [5,6,7]])
    @pytest.mark.parametrize("input_type", [np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_logistic(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)
