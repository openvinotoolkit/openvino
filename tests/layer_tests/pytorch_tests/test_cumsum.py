# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCumSum(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, axis, dtype_str):
        import torch

        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "uint8": torch.uint8
        }

        dtype = dtypes.get(dtype_str)

        class aten_cumsum(torch.nn.Module):
            def __init__(self, axis, dtype):
                super(aten_cumsum, self).__init__()
                self.axis = axis
                self.dtype = dtype
                if self.dtype is not None:
                    self.forward = self.forward_dtype

            def forward(self, x):
                return torch.cumsum(x, self.axis)
            
            def forward_dtype(self, x):
                return torch.cumsum(x, self.axis, dtype=self.dtype)

        ref_net = None

        return aten_cumsum(axis, dtype), ref_net, "aten::cumsum"

    @pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cumsum(self, axis, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(axis, dtype), ie_device, precision, ir_version)
