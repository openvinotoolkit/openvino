# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestAminMax(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype=None, out=False):
        import numpy as np
        x = np.array(inputs).astype(dtype)
        if not out:
            return (x, )
        return (x, np.zeros_like(x).astype(dtype))

    def create_model(self, mode="", dtype=None, dim=None, keepdim=False):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        dtype = dtype_map.get(dtype)

        class aten_aminmax(torch.nn.Module):
            def __init__(self, mode, dtype, dim, keepdim):
                super().__init__()
                self.dtype = dtype
                self.dim = dim
                self.keepdim = keepdim
                if mode == "out":
                    self.forward = self.forward_out
                else:
                    self.forward = self.forward_default

            def forward_default(self, x):
                return torch.aminmax(x.to(self.dtype), dim=self.dim, keepdim=self.keepdim)

            def forward_out(self, x, y):
                y = y[0].to(self.dtype), y[1].to(self.dtype)
                return torch.aminmax(x.to(self.dtype), dim=self.dim, keepdim=self.keepdim, out=y), y

        model_class = aten_aminmax(mode, dtype, dim, keepdim)

        ref_net = None

        return model_class, ref_net, "aten::aminmax"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("mode,dtype", [
        ("", "float32"), ("", "float64"), ("", "int32"), ("", "int64"),
        ("out", "float32"), ("out", "float64"), ("out", "int32"), ("out", "int64")])
    @pytest.mark.parametrize("inputs", [[0, 1, 2, 3, 4, -1], 
                                        [-2, -1, 0, 1, 2, 3],
                                        [1, 2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("dim,keepdim", [(None, False),  # Test with default arguments
                                             (0, False),     # Test with dim provided and keepdim=False
                                             (0, True),      # Test with dim provided and keepdim=True
                                             (None, True)])  # Test with keepdim=True and dim not provided
    def test_aminmax(self, mode, dtype, inputs, ie_device, 
                     precision, ir_version, dim, keepdim):
        self._test(
            *self.create_model(mode, dtype, dim, keepdim),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            freeze_model=False,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype, "out": mode == "out"}        
        )