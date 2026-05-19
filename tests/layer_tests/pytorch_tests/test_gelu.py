# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGelu(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3),)

    def create_model(self, approximate):
        import torch
        import torch.nn.functional as F

        class aten_gelu(torch.nn.Module):
            def __init__(self, approximate='none'):
                super().__init__()
                self.approximate = approximate

            def forward(self, x):
                return F.gelu(x, approximate=self.approximate)


        return aten_gelu(approximate), "aten::gelu"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_glu(self, approximate, ie_device, precision, ir_version):
        self._test(*self.create_model(approximate), ie_device, precision, ir_version)
