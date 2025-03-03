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
        return []

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_iota():
            return jax.lax.iota(input_type,input_shape)
        
        return jax_iota, None, 'iota'
        

    @pytest.mark.parametrize("input_shape", [1,2,3,4,5,6,7,8,9,10])
    @pytest.mark.parametrize("input_type", [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_iota(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)