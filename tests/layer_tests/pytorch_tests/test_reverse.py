# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class aten_flip_single_dim(torch.nn.Module):
    """Model that flips tensor along a single dimension."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flip(x, dims=(self.dim,))


class aten_flip_multi_dim(torch.nn.Module):
    """Model that flips tensor along multiple dimensions."""
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, dims=self.dims)


class aten_flip_all_dims(torch.nn.Module):
    """Model that flips tensor along all dimensions."""
    def __init__(self, ndim):
        super().__init__()
        self.dims = tuple(range(ndim))

    def forward(self, x):
        return torch.flip(x, dims=self.dims)


class TestReverse(PytorchLayerTest):
    """Test suite for torch.flip operation (reverse)."""
    
    def _prepare_input(self, input_shape=None, dtype=np.float32):
        """Generate input data for testing.
        
        Args:
            input_shape: Shape of input tensor. If None, uses self.input_shape
            dtype: Data type of input tensor
            
        Returns:
            Tuple containing the input numpy array
        """
        if input_shape is None:
            input_shape = self.input_shape
        
        if dtype in [np.float32, np.float64]:
            # Use sequential values for easier debugging
            return (np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape),)
        elif dtype in [np.int32, np.int64]:
            # Use modulo to keep values reasonable
            return (
                np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape) % 100,
            )
        elif dtype == np.bool_:
            # Random boolean values
            return (np.random.choice([True, False], size=input_shape).astype(dtype),)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def create_model(self, dims, single_dim=False, ndim=None):
        """Create a torch.flip model.
        
        Args:
            dims: Dimensions to flip (int, list of ints, or "all")
            single_dim: If True, use single dimension model
            ndim: Number of dimensions (used when dims="all")
            
        Returns:
            Tuple of (model, ref_net, op_name)
        """
        if single_dim:
            if isinstance(dims, int):
                dim_value = dims
            else:
                dim_value = dims[0]
            model = aten_flip_single_dim(dim_value)
        elif dims == "all":
            model = aten_flip_all_dims(ndim)
        else:
            model = aten_flip_multi_dim(dims)
        
        ref_net = None
        op_name = "aten::flip"
        return model, ref_net, op_name

    @pytest.mark.parametrize(
        "input_shape",
        [
            [5],              # 1D
            [3, 4],           # 2D
            [2, 3, 4],        # 3D
            [2, 3, 4, 5],     # 4D
            [1, 2, 3, 4, 5],  # 5D
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int32,
            np.int64,
            np.bool_,
        ],
    )
    @pytest.mark.parametrize(
        "dim_to_flip",
        [
            0,   # First dimension
            -1,  # Last dimension
            1,   # Second dimension
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_reverse_single_dim(
        self, ie_device, precision, ir_version, input_shape, dtype, dim_to_flip
    ):
        """Test reversing a single dimension with various shapes and data types."""
        # Skip invalid dimension indices
        if dim_to_flip < -len(input_shape) or dim_to_flip >= len(input_shape):
            pytest.skip(f"Dimension {dim_to_flip} invalid for shape {input_shape}")

        self.input_shape = input_shape
        # Normalize negative indices
        normalized_dim = (
            dim_to_flip if dim_to_flip >= 0 else len(input_shape) + dim_to_flip
        )
        
        self._test(
            *self.create_model([normalized_dim], single_dim=True),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            [3, 4, 5],        # 3D
            [2, 3, 4, 5],     # 4D
            [1, 2, 3, 4, 5],  # 5D
        ],
    )
    @pytest.mark.parametrize(
        "dims_to_flip",
        [
            [0, 1],       # First two dimensions
            [1, 2],       # Middle dimensions
            [0, -1],      # First and last
            [0, 1, 2],    # First three dimensions
            [-1, -2],     # Last two dimensions (negative indices)
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.int32,
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_reverse_multi_dim(
        self, ie_device, precision, ir_version, input_shape, dims_to_flip, dtype
    ):
        """Test reversing multiple dimensions simultaneously."""
        # Skip invalid dimension indices
        if any(d >= len(input_shape) or d < -len(input_shape) for d in dims_to_flip):
            pytest.skip(f"Dimensions {dims_to_flip} invalid for shape {input_shape}")

        # Normalize negative indices
        normalized_dims = [d if d >= 0 else len(input_shape) + d for d in dims_to_flip]
        
        self.input_shape = input_shape
        self._test(
            *self.create_model(normalized_dims),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            [5],          # 1D
            [3, 4],       # 2D
            [2, 3, 4],    # 3D
            [2, 3, 4, 5], # 4D
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.int64,
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_reverse_all_dims(self, ie_device, precision, ir_version, input_shape, dtype):
        """Test reversing all dimensions of a tensor."""
        self.input_shape = input_shape
        self._test(
            *self.create_model("all", ndim=len(input_shape)),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
        )

    @pytest.mark.parametrize(
        "input_shape",
        [
            [1, 5, 1, 3],  # Mixed dimensions with ones
            [1],           # Single element 1D
            [10],          # 1D array
            [1, 1],        # 2D with size 1
            [1, 1, 1, 1],  # 4D with all size 1
        ],
    )
    @pytest.mark.nightly
    def test_reverse_edge_cases(self, ie_device, precision, ir_version, input_shape):
        """Test edge cases with singleton dimensions and small tensors."""
        self.input_shape = input_shape
        # Test reversing dimension 0 for edge cases
        self._test(
            *self.create_model([0], single_dim=True),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": np.float32},
        )

    @pytest.mark.parametrize(
        "input_shape,dims_to_flip",
        [
            ([10, 20], [0]),         # Large 2D, flip rows
            ([10, 20], [-1]),         # Large 2D, flip columns
            ([5, 10, 15], [1]),      # Large 3D
            ([4, 8, 12, 16], [0, 2]), # Large 4D, multiple axes
        ],
    )
    @pytest.mark.nightly
    def test_reverse_large_tensors(
        self, ie_device, precision, ir_version, input_shape, dims_to_flip
    ):
        """Test reverse operation on larger tensors."""
        normalized_dims = [d if d >= 0 else len(input_shape) + d for d in dims_to_flip]
        self.input_shape = input_shape
        
        self._test(
            *self.create_model(normalized_dims),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": np.float32},
        )

    @pytest.mark.parametrize(
        "input_shape,dim_to_flip",
        [
            ([3, 4, 5, 6], -1),  # Last dimension (negative index)
            ([3, 4, 5, 6], -2),  # Second to last
            ([3, 4, 5, 6], -3),  # Third to last
            ([3, 4, 5, 6], -4),  # First dimension (via negative)
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_negative_indices(
        self, ie_device, precision, ir_version, input_shape, dim_to_flip
    ):
        """Test that negative dimension indices work correctly."""
        self.input_shape = input_shape
        normalized_dim = len(input_shape) + dim_to_flip
        
        self._test(
            *self.create_model([normalized_dim], single_dim=True),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": np.float32},
        )
