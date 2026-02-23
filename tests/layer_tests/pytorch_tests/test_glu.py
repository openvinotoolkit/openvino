# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGlu(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 4, 224, 224),)

    def create_model(self, dim):
        import torch
        import torch.nn.functional as F

        class aten_glu(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return F.glu(x, self.dim)


        return aten_glu(dim), "aten::glu"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2])
    def test_glu(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version)
