# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest



class TestSliceScatter(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4),)

    def create_model(self, src, dim, start, end, step):
        class aten_slice_scatter(torch.nn.Module):
            def __init__(self, src=None, dim=None, start=None, end=None, step=None):
                super().__init__()
                self.src = src
                self.dim = dim
                self.start = start
                self.end = end
                self.step = step

            def forward(self, x):
                return torch.slice_scatter(x, src=self.src, dim=self.dim, start=self.start, end=self.end, step=self.step)



        return aten_slice_scatter(src, dim, start, end, step), "aten::slice_scatter"

    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize(
        ("src", "dim", "start", "end", "step"),
        [
            (torch.ones(2, 1, 3, 4), 1, 1, 2, 1),
            (torch.ones(2, 5, 3, 4), 0, 0, 2, 1),
            (torch.ones(2, 5, 2, 4), 2, 0, 3, 2),
            (torch.ones(2, 5, 3, 3), 3, 1, 4, 1),
            (torch.ones(2, 3, 3, 4), 1, 0, 5, 2),
        ],
    )
    def test_slice_scatter(self, src, dim, start, end, step, ie_device, precision, ir_version):
        self._test(*self.create_model(src, dim, start, end, step),
                   ie_device, precision, ir_version)
