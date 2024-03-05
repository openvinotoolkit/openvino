# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestElu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3).astype(np.float32),)

    def create_model(self, alpha, inplace):
        import torch
        import torch.nn.functional as F

        class aten_elu(torch.nn.Module):
            def __init__(self, alpha, inplace):
                super(aten_elu, self).__init__()
                self.alpha = alpha
                self.inplace = inplace

            def forward(self, x):
                return F.elu(x, alpha=self.alpha, inplace=self.inplace)

        ref_net = None

        return aten_elu(alpha, inplace), ref_net, "aten::elu"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    @pytest.mark.parametrize("inplace", [True, False])
    def test_elu(self, alpha, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(alpha, inplace), ie_device, precision, ir_version)
