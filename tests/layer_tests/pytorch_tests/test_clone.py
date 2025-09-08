# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestClone(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self):
        import torch

        class aten_clone(torch.nn.Module):

            def forward(self, x):
                return torch.clone(x)

        ref_net = None

        return aten_clone(), ref_net, "aten::clone"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_clone(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
