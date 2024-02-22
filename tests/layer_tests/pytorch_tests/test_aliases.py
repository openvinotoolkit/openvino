# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_alias(torch.nn.Module):
    def forward(self, x):
        x[:, 1, :, :] = 4.
        return x


class aten_loop_alias(torch.nn.Module):
    def forward(self, x):
        for i in range(2):
            x[:, i, :, :] = 4.
        return x


class TestAliases(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_alias(self, ie_device, precision, ir_version):
        self._test(aten_alias(), None, ["aten::slice",
                                        "aten::select",
                                        "aten::copy_"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_loop_alias(self, ie_device, precision, ir_version):
        self._test(aten_loop_alias(), None, ["aten::slice",
                                             "aten::select",
                                             "aten::copy_",
                                             "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False)
