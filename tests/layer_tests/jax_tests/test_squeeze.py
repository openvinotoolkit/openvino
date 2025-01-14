# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest


class TestSqueeze(JaxLayerTest):
    def _prepare_input(self):
        inp = jnp.array(np.random.rand(*self.input_shape).astype(np.float32))
        return [inp]

    def create_model(self, input_shape, dimensions):
        self.input_shape = input_shape

        def jax_squeeze(inp):
            out = lax.squeeze(inp, dimensions=dimensions)
            return out

        return jax_squeeze, None, 'squeeze'

    test_data_basic = [
        dict(input_shape=[1, 10, 1, 1], dimensions=[0]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[2]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[3]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[0, 2]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[0, 3]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[2, 3]),
        dict(input_shape=[1, 10, 1, 1], dimensions=[0, 2, 3]),
        dict(input_shape=[5, 1, 1, 5], dimensions=[1]),
        dict(input_shape=[5, 1, 1, 5], dimensions=[2]),
        dict(input_shape=[5, 1, 1, 5], dimensions=[1, 2]),
        dict(input_shape=[5, 1, 3, 1], dimensions=[1]),
        dict(input_shape=[5, 1, 3, 1], dimensions=[3]),
        dict(input_shape=[5, 1, 3, 1], dimensions=[1, 3]),
        dict(input_shape=[2, 1, 3, 4], dimensions=[1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_squeeze(self, ie_device, precision, ir_version, params):
        self._test(*self.create_model(**params),
                   ie_device, precision,
                   ir_version)
