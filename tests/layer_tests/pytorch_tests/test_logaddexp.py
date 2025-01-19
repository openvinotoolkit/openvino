# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestLogAddExp(PytorchLayerTest):
    def _prepare_input(self, input1, input2, dtype="float32"):
        """Prepare inputs for logaddexp testing"""
        return (np.array(input1).astype(dtype), np.array(input2).astype(dtype))

    def create_model(self, dtype=None):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
        }

        class LogAddExpModel(torch.nn.Module):
            def __init__(self, dtype=None):
                super(LogAddExpModel, self).__init__()
                self.dtype = dtype_map.get(dtype) if dtype else None

            def forward(self, x, y):
                if self.dtype:
                    x = x.to(self.dtype)
                    y = y.to(self.dtype)
                return torch.logaddexp(x, y)

        model_class = LogAddExpModel(dtype)
        ref_net = None

        return model_class, ref_net, "aten::logaddexp"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype",
        [
            "float32",
            "float64",
        ],
    )
    @pytest.mark.parametrize(
        "input1,input2",
        [
            # Basic cases
            (0.0, 0.0),        # log(exp(0) + exp(0)) = log(2)
            (1.0, 1.0),        # log(exp(1) + exp(1)) = log(2*e)
            (-1.0, -1.0),      # log(exp(-1) + exp(-1)) = log(2/e)
            
            # One large, one small number
            (100.0, 0.0),      # Tests handling of large differences
            (-100.0, 0.0),     # Tests handling of negative large differences
            
            # Both large numbers
            (100.0, 100.0),    # Tests numerical stability with large numbers
            (-100.0, -100.0),  # Tests numerical stability with large negative numbers
            
            # Numbers with different signs
            (1.0, -1.0),       # Tests mixed positive/negative
            (-1.0, 1.0),       # Tests mixed negative/positive
            
            # Near-zero cases
            (1e-7, 1e-7),      # Tests handling of very small numbers
            (-1e-7, -1e-7),    # Tests handling of very small negative numbers
        ],
    )
    def test_logaddexp_basic(self, dtype, input1, input2, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2, "dtype": dtype},
            rtol=1e-5  # Relative tolerance for floating point comparisons
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype",
        [
            "float32",
            "float64",
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (3,),           # 1D array
            (2, 3),         # 2D array
            (2, 2, 2),      # 3D array
            (1, 1),         # Broadcasting test
            (3, 1),         # Broadcasting test
            (1, 3),         # Broadcasting test
        ],
    )
    def test_logaddexp_shapes(self, dtype, shape, ie_device, precision, ir_version):
        # Generate random inputs within a reasonable range
        input1 = np.random.uniform(-10, 10, shape)
        input2 = np.random.uniform(-10, 10, shape)
        
        self._test(
            *self.create_model(dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2, "dtype": dtype},
            rtol=1e-5
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_logaddexp_broadcasting(self, ie_device, precision, ir_version):
        # Test broadcasting with different shapes
        input1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape (1, 3)
        input2 = np.array([[1.0], [2.0]], dtype=np.float32)     # Shape (2, 1)
        
        self._test(
            *self.create_model("float32"),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2, "dtype": "float32"},
            rtol=1e-5
        )