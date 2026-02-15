# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestAngle(PytorchLayerTest):
    def _prepare_input(self, complex_type):
        shape = (5, 6, 7)
        if complex_type:
            shape += (2,)
        return [np.random.uniform(-10, 10, shape).astype(np.float32)]

    def create_model(self, complex_type):
        import torch

        class aten_angle(torch.nn.Module):
            def __init__(self, complex_type):
                super().__init__()
                self.complex_type = complex_type

            def forward(self, x):
                if self.complex_type:
                    x = torch.view_as_complex(x)
                res = torch.angle(x)
                return res

        return aten_angle(complex_type), None, "aten::angle"

    @pytest.mark.parametrize("complex_type", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_angle(self, complex_type, ie_device, precision, ir_version):
        self._test(*self.create_model(complex_type),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"complex_type": complex_type})
