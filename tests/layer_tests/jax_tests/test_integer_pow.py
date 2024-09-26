# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(34455)


class TestIntegerPow(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            x = rng.uniform(-3.0, 3.0, self.x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            x = rng.integers(-3, 3, self.x_shape).astype(self.input_type)
        else:
            x = rng.integers(0, 3, self.x_shape).astype(self.input_type)
        x = jnp.array(x)
        return [x]

    def create_model(self, x_shape, y, input_type):
        self.x_shape = x_shape
        self.input_type = input_type

        def jax_integer_pow(x):
            return jax.lax.integer_pow(x, y)

        return jax_integer_pow, None, 'integer_pow'

    @pytest.mark.parametrize('x_shape', [[3], [2, 3], [1, 2, 3]])
    @pytest.mark.parametrize('y', [2, 3, 4])
    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_integer_pow(self, x_shape, y, input_type,
                         ie_device, precision, ir_version):
        kwargs = {}
        if input_type == np.float16:
            kwargs["custom_eps"] = 2e-2
        self._test(*self.create_model(x_shape, y, input_type),
                   ie_device, precision, ir_version, **kwargs)
