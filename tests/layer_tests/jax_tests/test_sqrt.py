# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(34455)


class TestSqrt(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            x = rng.uniform(0, 8.0, self.input_shape).astype(self.input_type)
        else:
            x = rng.integers(0, 8, self.input_shape).astype(self.input_type)

        x = jnp.array(x)
        return [x]

    def create_model(self, input_shape, input_type):
        self.input_shape = input_shape
        self.input_type = input_type

        def jax_sqrt(x):
            return jax.numpy.sqrt(x)

        return jax_sqrt, None, 'sqrt'

    @pytest.mark.parametrize("input_shape", [[2], [3, 4]])
    @pytest.mark.parametrize("input_type", [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_sqrt(self, ie_device, precision, ir_version, input_shape, input_type):
        self._test(*self.create_model(input_shape, input_type),
                   ie_device, precision,
                   ir_version)
