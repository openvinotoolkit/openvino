# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSigmoid(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, inplace=False):
        import torch
        import torch.nn.functional as F

        class aten_sigmoid(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_sigmoid, self).__init__()
                self.op = torch.sigmoid if not inplace else torch.sigmoid_

            def forward(self, x):
                return x, self.op(x)

        ref_net = None

        return aten_sigmoid(inplace), ref_net, "aten::sigmoid" if not inplace else "aten::sigmoid_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("inplace", [True, False])
    def test_sigmoid(self, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version)