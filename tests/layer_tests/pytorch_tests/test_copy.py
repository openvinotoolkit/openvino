# Copyright (C) 2018-2025 Intel Corporation
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
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("value", [1, [2.5], range(224)])
    def test_copy_(self, value, ie_device, precision, ir_version):
        self._test(*self.create_model(value), ie_device, precision, ir_version)


class TestAliasCopy(PytorchLayerTest):
    def _prepare_input(self, out):
        import numpy as np
        if not out:
            return (np.random.randn(1, 3, 224, 224).astype(np.float32),)
        return (np.random.randn(1, 3, 224, 224).astype(np.float32), np.zeros((1, 3, 224, 224), dtype=np.float32))

    def create_model(self, out):
        import torch

        class aten_copy(torch.nn.Module):
            def __init__(self, out):
                super(aten_copy, self).__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.alias_copy(x)

            def forward_out(self, x, y):
                return torch.alias_copy(x, out=y), y

        ref_net = None

        return aten_copy(out), ref_net, "aten::alias_copy"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [True, False])
    def test_copy_(self, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out})
