# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random
import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestReshape(PytorchLayerTest):
    def _prepare_input(self, complex_type):
        shape = (1, 12, 12, 24)
        if complex_type:
            shape += (2,)
        return (np.random.uniform(0, 50, shape).astype(np.float32))

    def create_model(self, shape, complex_type):
        import torch

        class aten_reshape(torch.nn.Module):
            def __init__(self, shape, complex_type):
                super().__init__()
                self.shape = shape
                self.complex_type = complex_type

            def forward(self, x):
                if self.complex_type:
                    x = torch.view_as_complex(x)
                res = torch.reshape(x, self.shape)
                if self.complex_type:
                    res = torch.view_as_real(res)
                return res

        return aten_reshape(shape, complex_type), None, "aten::reshape"

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
    @pytest.mark.parametrize("complex_type", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_reshape(self, shape, complex_type, ie_device, precision, ir_version):
        self._test(*self.create_model(shape, complex_type),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"complex_type": complex_type})


class TestDynamicReshape(PytorchLayerTest):
    def _prepare_input(self):
        last_dym = random.randint(1, 2)
        return (np.random.uniform(0, 50, (1, 12, 12, 24)).astype(np.float32), last_dym)

    def create_model(self, shape):
        import torch

        class aten_reshape(torch.nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x, dym):
                dym2 = int(torch.ops.aten.sym_size(x, 3)/dym)
                return torch.reshape(x, [12, 12, dym2, dym])

        return aten_reshape(shape), None, "aten::reshape"

    @pytest.mark.parametrize(("shape"), [
        [12, 12, 24, 1],
        [12, 12, 12, 2],
        [24, 12, 12, 1],
    ])
    @pytest.mark.precommit_fx_backend
    def test_dynamic_reshape(self, shape, ie_device, precision, ir_version):
        self._test(*self.create_model(shape), ie_device, precision, ir_version)
