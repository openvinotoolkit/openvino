# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestRReLU(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, lower, upper, inplace):
        import torch
        import torch.nn.functional as F

        class aten_rrelu(torch.nn.Module):
            def __init__(self, lower=0.125, upper=0.3333333333333333, inplace=False):
                super(aten_rrelu, self).__init__()
                if lower is not None:
                    self.lower = lower
                else:
                    self.lower = 0.125
                if upper is not None:
                    self.upper = upper
                else:
                    self.upper = 0.3333333333333333
                self.inplace = inplace

            def forward(self, x):
                return x, F.rrelu(x, self.lower, self.upper, inplace=self.inplace, training=False)

        ref_net = None

        return aten_rrelu(lower, upper, inplace), ref_net, "aten::rrelu" if not inplace else "aten::rrelu_"

    @pytest.mark.parametrize("lower", [0.01, 0.1])
    @pytest.mark.parametrize("upper", [0.1, 0.5, None])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_rrelu(self, lower, upper, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(lower, upper, inplace), ie_device, precision, ir_version)
