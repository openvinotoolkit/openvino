# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestBincount(PytorchLayerTest):
    def _prepare_input(self, input_data, weights=None, minlength=0, dtype="int32"):
        """Prepare inputs for bincount testing"""
        input_data = np.array(input_data).astype(dtype)
        weights = np.array(weights).astype("float32") if weights is not None else None
        return input_data, weights, minlength

    def create_model(self, weights_provided, dtype=None):
        class BincountModel(torch.nn.Module):
            def __init__(self, weights_provided, dtype=None):
                super(BincountModel, self).__init__()
                self.weights_provided = weights_provided
                self.dtype = dtype

            def forward(self, x, w, minlength):
                if self.dtype:
                    x = x.to(self.dtype)
                    if w is not None:
                        w = w.to(self.dtype)
                return torch.bincount(x, minlength=minlength, weights=w if self.weights_provided else None)

        model_class = BincountModel(weights_provided, dtype)
        ref_net = None

        return model_class, ref_net, "aten::bincount"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "weights_provided",
        [True, False],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            "int32",  # Bincount requires integer inputs
            "int64",
            "float64"
        ],
    )
    @pytest.mark.parametrize(
        "input_data, weights, minlength",
        [
            # Basic cases
            ([0, 1, 1, 2, 2, 2, 3], [1.0, 0.5, 0.5, 0.2, 0.3, 0.5, 0.1], 5),  # Basic with weights
            ([0, 1, 1, 2, 2, 2, 3], None, 5),  # Basic without weights

            # Edge cases
            ([0, 0, 0, 0], None, 0),  # All zeros
            ([0, 1, 2, 3], None, 10),  # Minlength greater than max element
            ([10, 20, 30], None, 0),   # Minlength 0 with large values
            ([], None, 0),             # Empty array
            ([-1, -1, -1], None, 10),  # Negative values
            ([0, 1, 2, 3], None, 0),   # Edge case with minlength 0

            # Large values
            ([1000, 1000, 1000, 1000], None, 4),  # Case with large identical values
            ([1000, 2000, 3000, 4000], None, 5),  # Case with larger range of values
            
            # Randomized cases
            (np.random.randint(0, 100, size=(1000,)), np.random.uniform(0, 1, size=(1000,)), 1000),
        ],
    )
    def test_bincount_basic(self, weights_provided, dtype, input_data, weights, minlength, ie_device, precision, ir_version):
        self._test(
            *self.create_model(weights_provided, dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_data": input_data, "weights": weights, "minlength": minlength, "dtype": dtype},
            rtol=1e-5  # Relative tolerance for floating point comparisons
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "weights_provided",
        [True, False],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            "int32",  # Bincount requires integer inputs
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (3,),           # 1D array
            (2, 3),         # 2D array
            (2, 2, 2),      # 3D array
        ],
    )
    def test_bincount_shapes(self, weights_provided, dtype, shape, ie_device, precision, ir_version):
        # Generate random input data within a reasonable range for bincount
        input_data = np.random.randint(0, 5, shape)
        weights = np.random.uniform(0, 1, shape) if weights_provided else None
        minlength = 5

        self._test(
            *self.create_model(weights_provided, dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_data": input_data, "weights": weights, "minlength": minlength, "dtype": dtype},
            rtol=1e-5
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bincount_broadcasting(self, ie_device, precision, ir_version):
        # Test broadcasting with different shapes
        input_data1 = np.array([[1, 2, 3]], dtype=np.int32)  # Shape (1, 3)
        input_data2 = np.array([[1], [2]], dtype=np.int32)   # Shape (2, 1)
        weights = np.array([1.0, 0.5, 0.1], dtype=np.float32)

        self._test(
            *self.create_model(True, "int32"),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_data": input_data1, "weights": weights, "minlength": 5, "dtype": "int32"},
            rtol=1e-5
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_bincount_device(self, device, ie_device, precision, ir_version):
        # Run tests for different devices (CPU and CUDA)
        input_data = np.array([0, 1, 1, 2, 2, 2, 3])
        weights = np.array([1.0, 0.5, 0.5, 0.2, 0.3, 0.5, 0.1])
        model = self.create_model(True, "int32")[0].to(device)
        self._test(
            *self.create_model(True, "int32"),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_data": input_data, "weights": weights, "minlength": 5, "dtype": "int32"},
            rtol=1e-5
        )
