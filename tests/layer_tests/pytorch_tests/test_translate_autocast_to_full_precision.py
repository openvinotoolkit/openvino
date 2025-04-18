# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_autocast_to_full_precision(torch.nn.Module):
    def forward(self, x):
        return x.to(torch.float32)

class TestAutocastToFullPrecision(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (np.random.randn(2, 3, 224, 224).astype(dtype),)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_autocast_to_full_precision(self, dtype, ie_device, precision, ir_version):
        model = aten_autocast_to_full_precision()
        self._test(model, None, "aten::_autocast_to_full_precision", ie_device, 
                   precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})
