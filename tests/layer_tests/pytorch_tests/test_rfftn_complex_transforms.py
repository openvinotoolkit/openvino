# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestRFFTN(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.input_shape).astype(np.float32),)

    def create_model(self, dim, s, norm):
        class aten_fft_rfftn(torch.nn.Module):
            def __init__(self, dim, s, norm):
                super(aten_fft_rfftn, self).__init__()
                self.dim = dim
                self.s = s
                self.norm = norm

            def forward(self, x):
                rfftn = torch.fft.rfftn(x, s=self.s, dim=self.dim, norm=self.norm)
                r = rfftn.real
                i = rfftn.imag
                irfftn = torch.fft.irfftn(torch.complex(r, i), s=self.s, dim=self.dim, norm=self.norm)
                return irfftn, r, i

        ref_net = None

        return (
            aten_fft_rfftn(dim, s, norm),
            ref_net,
            ["aten::fft_irfftn", "aten::complex", "aten::fft_rfftn", "aten::real", "aten::imag"],
        )

    @pytest.mark.parametrize("input_shape", [[64, 49], [64, 50], [64, 64, 49]])
    @pytest.mark.parametrize("dim", [[0, -1], [-2, -1], None, [0, 1]])
    @pytest.mark.parametrize("s", [None, [-1, 49], [64, -1], [64, 49], [5, 1]])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho", None])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_rfftn(self, ie_device, precision, ir_version, input_shape, dim, s, norm):
        self.input_shape = input_shape
        # Unfrozen test would fail due to issues with prim::GetAttr containing lists, strings or none.
        self._test(*self.create_model(dim, s, norm), ie_device, precision, ir_version, custom_eps=1e-3,
                   freeze_model=True)
