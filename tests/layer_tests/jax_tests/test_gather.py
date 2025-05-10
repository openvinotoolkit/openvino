# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import numpy as jnp
from jax import lax

from jax_layer_test_class import JaxLayerTest

rng = np.random.default_rng(98765)


class TestGather(JaxLayerTest):
    def _prepare_input(self):
        data = rng.integers(-10, 10, self.data_shape).astype(self.input_type)
        indices = rng.integers(0, self.data_shape[self.axis], self.indices_shape).astype(np.int32)
        return [jnp.array(data), jnp.array(indices)]

    def create_model(self, data_shape, indices_shape, axis, input_type):
        self.data_shape = data_shape
        self.indices_shape = indices_shape
        self.axis = axis
        self.input_type = input_type

        def jax_gather(data, indices):
            rank = data.ndim

            dnums = lax.GatherDimensionNumbers(
                offset_dims=tuple(i for i in range(rank) if i != axis),
                collapsed_slice_dims=(axis,),
                start_index_map=(axis,)
            )

            slice_sizes = tuple(1 if i == axis else data.shape[i] for i in range(rank))
            indices = indices[..., None]
            result = lax.gather(data, indices, dimension_numbers=dnums, slice_sizes=slice_sizes)
            return result

        return jax_gather, None, 'gather'

    @pytest.mark.parametrize('data_shape,indices_shape,axis', [
        ([5], [3], 0),
        ([4, 6], [2, 3], 1),
        ([3, 4, 5], [2, 1], 2),
    ])
    @pytest.mark.parametrize('input_type', [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_gather(self, data_shape, indices_shape, axis, input_type,
                    ie_device, precision, ir_version):
        self._test(*self.create_model(data_shape, indices_shape, axis, input_type),
                   ie_device, precision, ir_version)
