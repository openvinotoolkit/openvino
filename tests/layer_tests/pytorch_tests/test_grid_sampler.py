# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestGridSampler(PytorchLayerTest):
    def _prepare_input(self,  h_in, w_in, h_out, w_out):
        import numpy as np
        return (np.random.randn(1, 3, h_in, w_in).astype(np.float32), np.random.randn(1, h_out, w_out, 2).astype(np.float32))

    def create_model(self, mode, padding_mode, align_corners):
        import torch
        import torch.nn.functional as F

        class aten_grid_sampler(torch.nn.Module):
            def __init__(self, mode, padding_mode, align_corners):
                super(aten_grid_sampler, self).__init__()
                self.mode = mode
                self.padding_mode = padding_mode
                self.align_corners = align_corners

            def forward(self, input, grid):
                return F.grid_sample(input, grid, self.mode, self.padding_mode, self.align_corners)

        ref_net = None

        return aten_grid_sampler(mode, padding_mode, align_corners), ref_net, "aten::grid_sampler"

    @pytest.mark.parametrize(["h_in", "w_in", "h_out", "w_out"], ([10, 10, 5, 5], [10, 15, 3, 5]))
    @pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
    @pytest.mark.parametrize("align_corners", [True, False, None])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_grid_sampler(self, h_in, w_in, h_out, w_out, mode, padding_mode, align_corners, ie_device, precision, ir_version):
        self._test(*self.create_model(mode, padding_mode, align_corners), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "h_in": h_in, "w_in": w_in, "h_out": h_out, "w_out": w_out
        })
