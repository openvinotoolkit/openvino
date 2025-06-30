# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestScalarTensor(PytorchLayerTest):

    def _prepare_input(self):
        return (np.array(np.random.randn(), dtype=np.float32),)

    def create_model(self):
        class aten_scalar_tensor(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, lhs):
                return torch.scalar_tensor(lhs.item())

        ref_net = None

        return aten_scalar_tensor(), ref_net, f"aten::scalar_tensor"

    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_scalar_tensor(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)


