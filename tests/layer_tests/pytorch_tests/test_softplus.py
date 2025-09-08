# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_softplus(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.softplus(x)


class TestSoftplus(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_softplus(self, ie_device, precision, ir_version):
        self._test(aten_softplus(), None, "aten::softplus",
                   ie_device, precision, ir_version)
