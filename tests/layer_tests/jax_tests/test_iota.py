# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(34455)


class TestIota(JaxLayerTest):

    def _prepare_input(self):
        # Ensure size is treated as a tuple if it's an integer
        size = (self.size,) if isinstance(self.size, int) else self.size

        # Generate random input using numpy and cast it to the correct type
        inp = jnp.array(np.random.rand(*size).astype(self.input_type))
        print(f"Generated input: {inp}")  # Debug statement
        return [inp]

    def create_model(self, input_type, size):
        self.input_type = input_type
        self.size = size
        def jax_iota(x):  # x is a placeholder, not used by iota
            return jax.lax.iota(self.input_type, self.size)
        return jax_iota, None, 'iota'
    


    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16, np.uint16,
                                            np.int32, np.uint32, np.int64, np.uint64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.parametrize('size', [2, 3, 4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_iota(self,input_type,size,
                         ie_device, precision, ir_version):
        kwargs = {}
        if input_type == np.float16:
            kwargs["custom_eps"] = 2e-2
        self._test(*self.create_model(input_type, size),
                   ie_device, precision, ir_version, **kwargs)
