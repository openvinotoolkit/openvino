# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestAddCMul(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, 3).astype(self.input_type),
                np.random.uniform(0, 50, 3).astype(self.input_type),
                np.random.uniform(0, 50, 3).astype(self.input_type))

    def create_model(self, value=None):
        import torch

        class aten_addcmul(torch.nn.Module):
            def __init__(self, value=None):
                super(aten_addcmul, self).__init__()
                self.value = value

            def forward(self, x, y, z):
                if self.value is not None:
                    return torch.addcmul(x, y, z, value=self.value)
                return torch.addcmul(x, y, z)

        ref_net = None

        return aten_addcmul(value), ref_net, "aten::addcmul"

    @pytest.mark.parametrize(("input_type", "value"), [
        [np.int32, None],
        [np.float32, None],
        [np.float64, None],
        [np.int32, 1],
        [np.int32, 2],
        [np.int32, 10],
        [np.int32, 110],
        [np.float32, 2.0],
        [np.float32, 3.123],
        [np.float32, 4.5],
        [np.float64, 41.5],
        [np.float64, 24.5],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_addcmul(self, input_type, value, ie_device, precision, ir_version):
        self.input_type = input_type
        self._test(*self.create_model(value), ie_device, precision, ir_version)
