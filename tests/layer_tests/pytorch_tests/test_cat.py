# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_cat(torch.nn.Module):
    def forward(self, x):
        return torch.cat([x, x], 1)


class aten_append_cat(torch.nn.Module):
    def forward(self, x):
        list = []
        list.append(x)
        list.append(x)
        return torch.cat(list, 1)

class aten_loop_append_cat(torch.nn.Module):
    def forward(self, x):
        list = []
        for i in range(3):
            list.append(x)
        return torch.cat(list, 1)


class TestCat(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 1, 3),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cat(self, ie_device, precision, ir_version):
        self._test(aten_cat(), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_append_cat(self, ie_device, precision, ir_version):
        self._test(aten_append_cat(), None, ["aten::cat", "aten::append", "prim::ListConstruct"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_loop_append_cat(self, ie_device, precision, ir_version):
        self._test(aten_loop_append_cat(), None, ["aten::cat", "aten::append", "prim::ListConstruct", "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False)
