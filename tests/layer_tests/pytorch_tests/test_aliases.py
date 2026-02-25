# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_alias(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y[:, 1, :, :] = 4.
        return y


class aten_alias_tensor(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        n,c,h,w = x.shape
        ones = torch.ones([2,h,w]).to(x.dtype)
        y[:, 1:, :, :] = ones
        return y


class aten_loop_alias(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        for i in range(2):
            y[:, i, :, :] = 4.
        return y


class TestAliases(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_alias(self, ie_device, precision, ir_version):
        self._test(aten_alias(), None, ["aten::slice",
                                        "aten::select",
                                        "aten::copy_"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_alias_tensor(self, ie_device, precision, ir_version):
        self._test(aten_alias_tensor(), None, ["aten::slice",
                                               "aten::copy_"],
                   ie_device, precision, ir_version, freeze_model=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_loop_alias(self, ie_device, precision, ir_version):
        self._test(aten_loop_alias(), None, ["aten::slice",
                                             "aten::select",
                                             "aten::copy_",
                                             "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False)
