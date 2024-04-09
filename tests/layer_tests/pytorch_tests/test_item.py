# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize("dtype", [pytest.param(np.int32, marks=pytest.mark.xfail(reason="item returns scalar and for i32 it returns i64")),
                                   np.int64,
                                   np.float32,
                                   pytest.param(np.float64, marks=pytest.mark.xfail(reason="item returns scalar and for f64 it returns f32"))])
@pytest.mark.parametrize("shape", [[], [1], [1, 1, 1]])
class TestItem(PytorchLayerTest):
    def _prepare_input(self):
        return [np.random.randn(1).astype(self.dtype).reshape(self.shape)]

    def create_model(self):
        class aten_item(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return x.item()

        ref_net = None

        return aten_item(), ref_net, "aten::item"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_item(self, ie_device, precision, ir_version, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        # Dynamic shapes are not supported by Squeeze implementation
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, dynamic_shapes=False)
