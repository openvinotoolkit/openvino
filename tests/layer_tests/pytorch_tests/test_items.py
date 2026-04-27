# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_items_dict_input(torch.nn.Module):
    def forward(self, x):
        d = {0: x, 1: x + x, 2: 2 * x}
        return d.items()


class TestItems(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_items(self, ie_device, precision, ir_version):
        self._test(aten_items_dict_input(), "aten::items",
                   ie_device, precision, ir_version, use_convert_model=True)
