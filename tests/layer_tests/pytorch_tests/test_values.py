# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestValues(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def test_values_sort(self, ie_device, precision, ir_version):
        class SortModel(torch.nn.Module):
            def forward(self, x):
                return torch.sort(x)[0]

        self._test(SortModel(), None, "aten::values",
                ie_device, precision, ir_version)

    def test_values_topk(self, ie_device, precision, ir_version):
        class TopKModel(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 5)[0]

        self._test(TopKModel(), None, "aten::values",
                ie_device, precision, ir_version)
