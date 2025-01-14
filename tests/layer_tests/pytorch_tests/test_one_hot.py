# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestOneHot(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randint(0, 100, (1,1000)).astype(np.int32),)

    def create_model(self, num_classes):
        import torch
        import torch.nn.functional as F

        class aten_one_hot(torch.nn.Module):
            def __init__(self, num_classes):
                super(aten_one_hot, self).__init__()
                self.num_classes = num_classes

            def forward(self, x):
                return F.one_hot(torch.arange(0, x.numel()) % 3, self.num_classes)

        return aten_one_hot(num_classes), None, "aten::one_hot"

    @pytest.mark.parametrize(("num_classes"), [-1, 3, 1000,])
    @pytest.mark.nightly
    #@pytest.mark.precommit
    def test_one_hot(self, num_classes, ie_device, precision, ir_version):
        self._test(*self.create_model(num_classes),
                   ie_device, precision, ir_version)
