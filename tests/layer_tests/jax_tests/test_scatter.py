# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from jax_layer_test_class import JaxLayerTest

class TestScatter(JaxLayerTest):
    def _prepare_input(self):
        operand = jnp.array(np.random.rand(*self.operand_shape).astype(np.float32))
        scatter_indices = jnp.array(np.random.randint([self.operand_shape[scatter_indices_dim] 
                                                    for scatter_indices_dim in self.scatter_dims_to_operand_dims], 
                                                    size=self.scatter_indices_shape))
        if len(scatter_indices.shape) == 1:
                scatter_indices = scatter_indices[:, None]
        updates = jnp.array(np.random.rand(*self.updates_shape).astype(np.float32))
        return (operand, scatter_indices, updates)
    

    def create_model(self, operand_shape, scatter_indices_shape, updates_shape, update_window_dims, 
                     inserted_window_dims, scatter_dims_to_operand_dims):
        self.operand_shape = operand_shape
        self.scatter_indices_shape = scatter_indices_shape
        self.scatter_dims_to_operand_dims = scatter_dims_to_operand_dims
        self.updates_shape = updates_shape

        def jax_scatter(operand, scatter_indices, updates):
            out = lax.scatter(operand=operand, scatter_indices=scatter_indices, updates=updates, 
                              dimension_numbers=lax.ScatterDimensionNumbers(
                                                update_window_dims=update_window_dims,
                                                inserted_window_dims=inserted_window_dims,
                                                scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
                                ),
            )
            return out
        
        return jax_scatter, None, 'scatter'

    
    test_data_basic = [
        dict(operand_shape=[64], scatter_indices_shape=[32], updates_shape=[32], 
            update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)),
        dict(operand_shape=[128, 64], scatter_indices_shape=[128, 2], updates_shape=[128], 
            update_window_dims=(), inserted_window_dims=(0,1,), scatter_dims_to_operand_dims=(0,1,)),
        dict(operand_shape=[128, 64], scatter_indices_shape=[32], updates_shape=[32, 128], 
            update_window_dims=(1,), inserted_window_dims=(1,), scatter_dims_to_operand_dims=(1,)),
        dict(operand_shape=[128, 64], scatter_indices_shape=[32], updates_shape=[32, 64], 
            update_window_dims=(1,), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)),
        dict(operand_shape=[128, 64, 32], scatter_indices_shape=[87], updates_shape=[87, 64, 32], 
            update_window_dims=(1,2,), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)),
        dict(operand_shape=[128, 64, 32], scatter_indices_shape=[55,2], updates_shape=[55, 64], 
            update_window_dims=(1,), inserted_window_dims=(0,2,), scatter_dims_to_operand_dims=(0,2)),
        dict(operand_shape=[64, 128, 32, 16], scatter_indices_shape=[440, 2], updates_shape=[440, 128, 16], 
            update_window_dims=(1,2,), inserted_window_dims=(0,2,), scatter_dims_to_operand_dims=(0,2,)),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    def test_scatter(self, ie_device, precision, ir_version, params):
        self._test(*self.create_model(**params), ie_device, precision,
                   ir_version)
