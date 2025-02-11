# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestIf(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32), self.y)

    def create_model(self):
        import torch
        import torch.nn.functional as F

        class prim_if(torch.nn.Module):
            def __init__(self):
                super(prim_if, self).__init__()

            def forward(self, x, y):
                if y > 0:
                    res = x.new_empty((0, 10), dtype=torch.uint8)
                else:
                    res = torch.zeros(x.shape[:2], dtype=torch.bool)
                return res.to(torch.bool)

        ref_net = None

        return prim_if(), ref_net, "prim::If"

    @pytest.mark.parametrize("y", [np.array(1),
                                   np.array(-1)
                                   ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == 'true', reason="Ticket - 114818")
    def test_if(self, y, ie_device, precision, ir_version):
        self.y = y
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)
