# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(56342)
class TestNeg(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            inp = rng.uniform(-5.0, 5.0, self.input_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inp = rng.integers(-8, 8, self.input_shape).astype(self.input_type)
        else:
            inp = rng.integers(0, 8, self.input_shape).astype(self.input_type)
        inp = jnp.array(inp)
        return [inp]

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_neg(inp):
            out = lax.neg(inp)
            return out

        return jax_neg, None, 'neg'

    @pytest.mark.parametrize("input_shape", [
        [10],
        [2, 3],
        [3, 4, 5],
        [7, 16, 11, 20],
        [6, 5, 4, 3, 2],
        [1, 1, 9, 1]
    ])
    @pytest.mark.parametrize('input_type', [np.int8, np.int16, np.int32,
                                            np.int64,np.float16, 
                                            np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_neg(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape=input_shape, input_type=input_type),
                   ie_device, precision,
                   ir_version)