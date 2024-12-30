# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(5402)


class TestIota(JaxLayerTest):
    def _prepare_input(self):
        return (self.input_type, self.input_shape)

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_iota(dtype, size):
            return jax.lax.iota(dtype, size)
        
        return jax_iota, None, 'iota'
        

    @pytest.mark.parametrize("input_shape", [1,2,3])
    @pytest.mark.parametrize("input_type", ["np.float32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_iota(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)