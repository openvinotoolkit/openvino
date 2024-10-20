# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(706670)


class TestArgmax(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            x = rng.uniform(-5.0, 5.0,
                            self.input_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            x = rng.integers(-8, 8, self.input_shape).astype(self.input_type)
        else:
            x = rng.integers(0, 8, self.input_shape).astype(self.input_type)

        if self.input_duplicate:
            x = np.concatenate((x, x), axis=self.axis)

        x = jnp.array(x)
        return [x]

    def create_model(self, input_shape, axis, input_type, index_dtype, input_duplicate):
        self.input_shape = input_shape
        self.axis = axis
        self.input_type = input_type
        self.input_duplicate = input_duplicate

        def jax_argmax(inp):
            out = lax.argmax(inp, axis, index_dtype)
            return out

        return jax_argmax, None, 'argmax'

    # Only [0, rank - 1] are valid axes for lax.argmax
    @pytest.mark.parametrize('input_shape, axis', [([64], 0),
                                                   ([64, 16], 0),
                                                   ([64, 16], 1),
                                                   ([48, 23, 54], 0),
                                                   ([48, 23, 54], 1),
                                                   ([48, 23, 54], 2),
                                                   ([2, 18, 32, 25], 0),
                                                   ([2, 18, 32, 25], 1),
                                                   ([2, 18, 32, 25], 2),
                                                   ([2, 18, 32, 25], 3)])
    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("input_duplicate", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit_jax_fe
    def test_argmax(self, ie_device, precision, ir_version, input_shape, axis, input_type, index_dtype, input_duplicate):
        self._test(*self.create_model(input_shape, axis, input_type, index_dtype, input_duplicate),
                   ie_device, precision, ir_version)
