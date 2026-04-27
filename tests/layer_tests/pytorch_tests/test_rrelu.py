# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestRReLU(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (self.random.randn(1, 3, 224, 224),)

    def create_model(self, lower, upper, training, inplace):
        import torch
        import torch.nn.functional as F

        class AtenRReLU(torch.nn.Module):
            def __init__(self, lower=1 / 8, upper=1 / 3, training=False, inplace=False):
                super(AtenRReLU, self).__init__()
                self.lower = lower
                self.upper = upper
                self.training_flag = training
                self.inplace = inplace

            def forward(self, x):
                # Call F.rrelu with different argument combinations depending
                # on which bounds were provided, to cover converter branches.
                if self.lower is None and self.upper is None:
                    out = F.rrelu(x, training=self.training_flag, inplace=self.inplace)
                elif self.lower is None:
                    out = F.rrelu(x, upper=self.upper, training=self.training_flag, inplace=self.inplace)
                elif self.upper is None:
                    out = F.rrelu(x, self.lower, training=self.training_flag, inplace=self.inplace)
                else:
                    out = F.rrelu(x, self.lower, self.upper, training=self.training_flag, inplace=self.inplace)
                return x, out

        return AtenRReLU(lower, upper, training, inplace), "aten::rrelu" if not inplace else "aten::rrelu_"

    @pytest.mark.parametrize(
        "lower,upper",
        [
            (None, None),  # default bounds
            (None, 0.5),  # only upper bound provided
            (0.1, None),  # only lower bound provided
            (0.125, 0.333),  # both bounds provided
        ],
    )
    @pytest.mark.parametrize("training", [False])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_rrelu(self, lower, upper, training, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(lower, upper, training, inplace), ie_device, precision, ir_version)
