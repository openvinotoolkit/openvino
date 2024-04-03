# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestAmin(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3).astype(np.float32),)

    def create_model(self):
        class aten_aminmax(torch.nn.Module):
            def __init__(self):
                super(aten_aminmax, self).__init__()

            def forward(self, x):
                amin, amax = x.min(), x.max()
                return amin, amax

        ref_net = None

        return aten_aminmax(), ref_net, "aten::aminmax"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_aminmax(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)