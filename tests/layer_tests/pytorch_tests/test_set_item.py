# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import platform

from pytorch_layer_test_class import PytorchLayerTest


class TestSetItem(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return [np.random.randint(-10, 10, [10]).tolist()]

    def create_model(self, idx):
        import torch
        from typing import List

        class aten_set_item(torch.nn.Module):
            def __init__(self, idx):
                super(aten_set_item, self).__init__()
                self.idx = idx

            def forward(self, x: List[int]):
                x[self.idx] = 0
                return torch.tensor(x).to(torch.int)

        ref_net = None

        return aten_set_item(idx), ref_net, "aten::_set_item"

    @pytest.mark.parametrize("idx", [0, 1, -1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_set_item_list(self, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(idx), ie_device, precision, ir_version)
