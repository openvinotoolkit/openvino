# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRoll(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (2, 3, 4)).astype(np.float32),)

    def create_model(self, shifts, dim):
        import torch

        class aten_roll(torch.nn.Module):
            def __init__(self, shifts, dim=None):
                super(aten_roll, self).__init__()
                self.dim = dim
                self.shifts = shifts

            def forward(self, x):
                if self.dim is not None:
                    return torch.roll(x, self.shifts, self.dim)
                return torch.roll(x, self.shifts)

        ref_net = None

        return aten_roll(shifts, dim), ref_net, "aten::roll"

    @pytest.mark.parametrize(("shifts", "dim"), [
        [(2, 1), (0, 1)],
        [1, 0],
        [-1, 0],
        [1, None],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_roll(self, shifts, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(shifts, dim), ie_device, precision, ir_version,
                   dynamic_shapes=ie_device != "GPU")
