# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(34455)


class TestConcatenate(JaxLayerTest):
    def _prepare_input(self):
        inputs = []
        for input_shape in self.input_shapes:
            if np.issubdtype(self.input_type, np.floating):
                x = rng.uniform(-5.0, 5.0, input_shape).astype(self.input_type)
            elif np.issubdtype(self.input_type, np.signedinteger):
                x = rng.integers(-8, 8, input_shape).astype(self.input_type)
            else:
                x = rng.integers(0, 8, input_shape).astype(self.input_type)
            x = jnp.array(x)
            inputs.append(x)
        return inputs

    def create_model(self, input_shapes, input_type, dimension):
        self.input_shapes = input_shapes
        self.input_type = input_type

        def jax_concatenate(*arrays):
            return jax.lax.concatenate(arrays, dimension)

        return jax_concatenate, None, 'concatenate'

    @pytest.mark.parametrize('input_shapes,dimension', [
        ([[2], [3]], 0),
        ([[2, 3], [2, 4]], 1),
        ([[1, 2, 3], [1, 2, 2], [1, 2, 1]], 2),
    ])
    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_concatenate(self, input_shapes, input_type, dimension,
                         ie_device, precision, ir_version):
        self._test(*self.create_model(input_shapes, input_type, dimension),
                   ie_device, precision, ir_version)
