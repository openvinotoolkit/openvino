# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestLogAddExp(PytorchLayerTest):
    def _prepare_input(self, input1, input2, dtype1="float32", dtype2="float32", out=False):
        """Prepare inputs for logaddexp testing"""
        inp1 = np.array(input1).astype(dtype1)
        inp2 = np.array(input2).astype(dtype2)
        if not out:
            return (inp1, inp2)
        return (inp1, inp2, np.zeros_like(np.logaddexp(inp1, inp2)))

    def create_model(self, out=False):
        import torch


        class LogAddExpModel(torch.nn.Module):
            def __init__(self, out):
                super(LogAddExpModel, self).__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x, y):
                return torch.logaddexp(x, y)
        
            def forward_out(self, x, y, out):
                return torch.logaddexp(x, y, out=out), out

        model_class = LogAddExpModel(out)
        ref_net = None

        return model_class, ref_net, "aten::logaddexp"

    @pytest.mark.parametrize(
        "dtype1",
        [
            "float32",
            "float64",
        ],
    )
    @pytest.mark.parametrize(
        "dtype2",
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
    @pytest.mark.parametrize(
        "out", (False, True)
    )
    def test_logaddexp_basic(self, dtype1, dtype2, input1, input2, out, ie_device, precision, ir_version):
        self._test(
            *self.create_model(out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2, "dtype1": dtype1, "dtype2": dtype2, "out": out},
            rtol=1e-5  # Relative tolerance for floating point comparisons
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype1",
        [
            "float32",
            "float64",
        ],
    )
    @pytest.mark.parametrize(
        "dtype2",
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
    @pytest.mark.parametrize(
        "out", (False, True)
    )
    def test_logaddexp_shapes(self, dtype1, dtype2, shape, out, ie_device, precision, ir_version):
        # Generate random inputs within a reasonable range
        input1 = np.random.uniform(-10, 10, shape)
        input2 = np.random.uniform(-10, 10, shape)
        
        self._test(
            *self.create_model(out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2, "dtype1": dtype1, "dtype2": dtype2, "out": out},
            rtol=1e-5
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_logaddexp_broadcasting(self, ie_device, precision, ir_version):
        # Test broadcasting with different shapes
        input1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape (1, 3)
        input2 = np.array([[1.0], [2.0]], dtype=np.float32)     # Shape (2, 1)
        
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input1": input1, "input2": input2},
            rtol=1e-5
        )