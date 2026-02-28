# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestDelete(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(2, 3).astype(np.float32),)

    class ListDeleteModel(torch.nn.Module):
        def forward(self, x):
            l = [x, x + 1.0]
            del l[-1]
            return l[0]

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_delete(self, ie_device, precision, ir_version):
        self._test(self.ListDeleteModel(), None, ie_device, precision, ir_version)
