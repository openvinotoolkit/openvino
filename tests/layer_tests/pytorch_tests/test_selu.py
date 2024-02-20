# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSilu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, inplace=False):
        import torch
        import torch.nn.functional as F

        class aten_selu(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_selu, self).__init__()
                self.inplace = inplace

            def forward(self, x):
                return x, F.selu(x, inplace=self.inplace)

        ref_net = None

        return aten_selu(inplace), ref_net, "aten::selu" if not inplace else "aten::selu_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("inplace", [True, False])
    def test_silu(self, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version)
