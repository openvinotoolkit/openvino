# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestNdtr(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), dtype=np.float32, out_dtype=None):
        if np.issubdtype(dtype, np.integer):
            x = self.random.randint(-10, 10, size=input_shape, dtype=dtype)
        else:
            x = self.random.uniform(-5.0, 5.0, size=input_shape, dtype=dtype)
        if out_dtype is None:
            return (x,)
        return (x, np.zeros(input_shape, dtype=out_dtype))

    def create_model(self, out=False):
        class aten_special_ndtr(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.special.ndtr(x)

            def forward_out(self, x, y):
                return torch.special.ndtr(x, out=y), y

        return aten_special_ndtr(out), "aten::special_ndtr"

    @pytest.mark.parametrize("out_dtype", [None, np.float16, np.float32])
    @pytest.mark.parametrize("input_shape", [(3, 3), (2, 3, 4)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_ndtr(self, out_dtype, input_shape, dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model(out=out_dtype is not None),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input_shape": input_shape,
                "dtype": dtype,
                "out_dtype": out_dtype,
            },
        )
