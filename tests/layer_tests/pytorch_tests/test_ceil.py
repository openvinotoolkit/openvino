# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCeil(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, inplace):
        import torch

        class aten_ceil(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_ceil, self).__init__()
                self.op = torch.ceil_ if inplace else torch.ceil

            def forward(self, x):
                return x, self.op(x)

        ref_net = None

        return aten_ceil(inplace), ref_net, "aten::ceil" if not inplace else "aten::ceil_"

    @pytest.mark.parametrize("inplace", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ceil(self, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(inplace), ie_device, precision, ir_version)
