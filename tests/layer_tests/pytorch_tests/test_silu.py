# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSilu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self):
        import torch
        import torch.nn.functional as F

        class aten_silu(torch.nn.Module):
            def __init__(self):
                super(aten_silu, self).__init__()

            def forward(self, x):
                return F.silu(x)

        ref_net = None

        return aten_silu(), ref_net, "aten::silu"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_silu(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
