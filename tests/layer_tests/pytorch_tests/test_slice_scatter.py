# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest



class TestSliceScatter(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self, src, dim, start, end, step):

        import torch
        class aten_slice_scatter(torch.nn.Module):
            def __init__(self, src=None, dim=None, start=None, end=None, step=None):
                super(aten_slice_scatter, self).__init__()
                self.src = src
                self.dim = dim
                self.start = start
                self.end = end
                self.step = step

            def forward(self, x):
                return torch.slice_scatter(x, src=self.src, dim=self.dim, start=self.start, end=self.end, step=self.step);


        ref_net = None

        return aten_slice_scatter(src, dim, start, end, step), ref_net, "aten::slice_scatter"

    import torch
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize(("src", "dim", "start", "end", "step"),
                             [(torch.ones(2), 1, 1, 2, 1),])
    def aten_slice_scatter(self, src, dim, start, end, step, ie_device, precision, ir_version):
        self._test(*self.create_model(src, dim, start, end, step),
                   ie_device, precision, ir_version)
