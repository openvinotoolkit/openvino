# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(56342)

class TestBinaryComparison(JaxLayerTest):
    def _prepare_input(self):
        if np.issubdtype(self.input_type, np.floating):
            x = rng.uniform(-5.0, 5.0, self.input_shapes[0]).astype(self.input_type)
            y = rng.uniform(-5.0, 5.0, self.input_shapes[1]).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            x = rng.integers(-8, 8, self.input_shapes[0]).astype(self.input_type)
            y = rng.integers(-8, 8, self.input_shapes[1]).astype(self.input_type)
        else:
            x = rng.integers(0, 8, self.input_shapes[0]).astype(self.input_type)
            y = rng.integers(0, 8, self.input_shapes[1]).astype(self.input_type)
        x = jnp.array(x)
        y = jnp.array(y)
        return [x, y]

    def create_model(self, input_shapes, binary_op, input_type):
        reduce_map = {
            'eq': lax.eq,
            'ge': lax.ge,
            'gt': lax.gt,
            'lt': lax.lt,
            'le': lax.le,
            'ne': lax.ne
        }

        self.input_shapes = input_shapes
        self.input_type = input_type
        
        def jax_binary(x, y):
            return reduce_map[binary_op](x, y)

        return jax_binary, None, binary_op

    @pytest.mark.parametrize('input_shapes', [[[5], [1]], [[1], [5]], [[2, 2, 4], [1, 1, 4]],
                                              [[5, 10], [5, 10]], [[2, 4, 6], [1, 4, 6]],
                                              [[5, 8, 10, 128], [5, 1, 10, 128]]])
    @pytest.mark.parametrize('binary_op', ['eq', 'ge', 'gt', 'lt', 'le','ne'])
    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_binary(self, input_shapes, binary_op, input_type,
                    ie_device, precision, ir_version):
        self._test(*self.create_model(input_shapes,  binary_op, input_type),
                   ie_device, precision, ir_version)
