# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp
import jax

from jax_layer_test_class import JaxLayerTest


class TestGather(JaxLayerTest):
    def _prepare_input(self):
        # Generate Operand
        if np.issubdtype(self.input_type, np.floating):
            operand = np.random.uniform(-10, 10, self.operand_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            operand = np.random.randint(-10, 10, self.operand_shape).astype(self.input_type)
        else:
            operand = np.random.randint(0, 10, self.operand_shape).astype(self.input_type)

        # Generate Indices
        # Logic untuk CLIP (mode 1) vs PROMISE_IN_BOUNDS (mode 0)
        limit = min(self.operand_shape)
        
        if self.mode == 1: # CLIP: Generate OOB indices untuk stress test
            # Range index agak ekstrim: negatif s/d melebihi batas
            start_indices = np.random.randint(-5, limit + 5, self.indices_shape).astype(np.int32)
        elif self.mode == 2: # FILL_OR_DROP: Juga boleh OOB
            start_indices = np.random.randint(-2, limit + 2, self.indices_shape).astype(np.int32)
        else: # PROMISE_IN_BOUNDS: Harus aman
            # Safe limit = min_dim - max_slice_size
            max_slice = max(self.slice_sizes) if self.slice_sizes else 0
            safe_limit = limit - max_slice
            if safe_limit < 1: safe_limit = 1
            start_indices = np.random.randint(0, safe_limit, self.indices_shape).astype(np.int32)

        return [jnp.array(operand), jnp.array(start_indices)]

    def create_model(self, operand_shape, indices_shape, dimension_numbers, slice_sizes, mode, input_type):
        self.operand_shape = operand_shape
        self.indices_shape = indices_shape
        self.slice_sizes = slice_sizes
        self.mode = mode
        self.input_type = input_type

        def jax_gather(operand, start_indices):
            # Mapping integer mode ke JAX Enum
            jax_mode = lax.GatherScatterMode.PROMISE_IN_BOUNDS
            if mode == 1:
                jax_mode = lax.GatherScatterMode.CLIP
            elif mode == 2:
                jax_mode = lax.GatherScatterMode.FILL_OR_DROP

            return lax.gather(operand, start_indices,
                              dimension_numbers=dimension_numbers,
                              slice_sizes=slice_sizes,
                              mode=jax_mode)

        return jax_gather, None, 'gather'

    test_data = [
        # Case 1: Point Gather Identity (Scalar slice, no reordering)
        dict(
            operand_shape=[10, 5],
            indices_shape=[3, 1],
            slice_sizes=(1, 5),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            mode=0
        ),
        # Case 2: Reordering Case (Transpose logic check)
        dict(
            operand_shape=[5, 5, 5],
            indices_shape=[2, 2],
            slice_sizes=(1, 1, 5),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0, 1),
                start_index_map=(1, 0) # start_index_map dibalik
            ),
            mode=0
        ),
        # Case 3: CLIP Mode (Logic Min/Max)
        dict(
            operand_shape=[10],
            indices_shape=[5, 1],
            slice_sizes=(1,),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            mode=1 # CLIP
        ),
        # Case 4: FILL_OR_DROP Mode (Logic Masking)
        dict(
            operand_shape=[10],
            indices_shape=[5, 1],
            slice_sizes=(1,),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            mode=2 # FILL_OR_DROP
        ),
        # Case 5: Batch Gather (Standard Use Case)
        dict(
            operand_shape=[16, 128, 512],
            indices_shape=[16, 1],
            slice_sizes=(1, 128, 512),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1, 2), # Semua dimensi dipertahankan (kecuali yang dicollapse)
                collapsed_slice_dims=(0,),
                start_index_map=(0,)
            ),
            mode=0
        ),
    ]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_jax_fe
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    def test_gather(self, ie_device, precision, ir_version, params, input_type):
        self._test(*self.create_model(**params, input_type=input_type),
                   ie_device, precision,
                   ir_version)
