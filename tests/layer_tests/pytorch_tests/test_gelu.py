# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3).astype(np.float32),)

    def create_model(self, approximate):
        import torch
        import torch.nn.functional as F

        class aten_gelu(torch.nn.Module):
            def __init__(self, approximate='none'):
                super(aten_gelu, self).__init__()
                self.approximate = approximate

            def forward(self, x):
                return F.gelu(x, approximate=self.approximate)

        ref_net = None

        return aten_gelu(approximate), ref_net, "aten::gelu"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_glu(self, approximate, ie_device, precision, ir_version):
        self._test(*self.create_model(approximate), ie_device, precision, ir_version)
