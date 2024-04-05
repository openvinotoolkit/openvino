# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestAminMax(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype=None):
        import numpy as np
        return [np.array(inputs).astype(dtype)]

    def create_model(self, dtype=None, dim=None, keepdim=False):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        dtype = dtype_map.get(dtype)

        class aten_aminmax(torch.nn.Module):
            def __init__(self, dtype, dim, keepdim):
                super().__init__()
                self.dtype = dtype
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.aminmax(x.to(self.dtype), dim=self.dim, keepdim=self.keepdim, out=None)

        model_class = aten_aminmax(dtype, dim, keepdim)

        ref_net = None

        return model_class, ref_net, "aten::aminmax"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    @pytest.mark.parametrize("inputs", [[0, 1, 2, 3, 4, -1],
                                        [-2, -1, 0, 1, 2, 3],
                                        [1, 2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("dim,keepdim", [(None, False),  # Test with default arguments
                                             (0, False),     # Test with dim provided and keepdim=False
                                             (0, True),      # Test with dim provided and keepdim=True
                                             (None, True)])  # Test with keepdim=True and dim not provided
    def test_aminmax(self, dtype, inputs, ie_device,
                     precision, ir_version, dim, keepdim):
        self._test(
            *self.create_model(dtype=dtype, dim=dim, keepdim=keepdim),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            freeze_model=False,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype}
        )
