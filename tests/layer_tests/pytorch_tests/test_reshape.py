# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import random

from pytorch_layer_test_class import PytorchLayerTest


class TestReshape(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (1, 12, 12, 24)).astype(np.float32))

    def create_model(self, shape):
        import torch

        class aten_reshape(torch.nn.Module):
            def __init__(self, shape):
                super(aten_reshape, self).__init__()
                self.shape = shape

            def forward(self, x):
                return torch.reshape(x, self.shape)

        ref_net = None

        return aten_reshape(shape), ref_net, "aten::reshape"

    @pytest.mark.parametrize(("shape"), [
        [-1, 6],
        [12, 12, 24, 1],
        [12, 12, 12, 2],
        [12, -1, 12, 24],
        [24, 12, 12, 1],
        [24, 12, 12, -1],
        [24, 1, -1, 12],
        [24, 1, 1, -1, 12],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_reshape(self, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version)

class TestDynamicReshape(PytorchLayerTest):
    def _prepare_input(self):
        last_dym = random.randint(1,2)
        return (np.random.uniform(0, 50, (1, 12, 12, 24)).astype(np.float32), last_dym)

    def create_model(self, shape):
        import torch

        class aten_reshape(torch.nn.Module):
            def __init__(self, shape):
                super(aten_reshape, self).__init__()
                self.shape = shape

            def forward(self, x, dym):
                #return torch.reshape(x, self.shape)
                dym2 = int(torch.ops.aten.sym_size(x, 3)/dym)
                return torch.reshape(x, [12, 12, dym2, dym])

        ref_net = None

        return aten_reshape(shape), ref_net, "aten::reshape"

    @pytest.mark.parametrize(("shape"), [
        [12, 12, 24, 1],
        [12, 12, 12, 2],
        [24, 12, 12, 1],
    ])
    @pytest.mark.precommit_fx_backend
    def test_dynamic_reshape(self, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version)
