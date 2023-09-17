# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCopy(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, value):
        import torch

        class aten_copy(torch.nn.Module):
            def __init__(self, value):
                super(aten_copy, self).__init__()
                self.value = torch.tensor(value)

            def forward(self, x):
                return x.copy_(self.value)

        ref_net = None

        return aten_copy(value), ref_net, "aten::copy_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("value", [1, [2.5], range(224)])
    def test_copy_(self, value, ie_device, precision, ir_version):
        self._test(*self.create_model(value), ie_device, precision, ir_version)
