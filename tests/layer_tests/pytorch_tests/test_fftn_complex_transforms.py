# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestFFTN(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.input_shape).astype(np.float32),)

    def create_model(self, dim, s, norm):
        class aten_fft_fftn(torch.nn.Module):
            def __init__(self, dim, s, norm):
                super(aten_fft_fftn, self).__init__()
                self.dim = dim
                self.s = s
                self.norm = norm

            def forward(self, x):
                fftn = torch.fft.fftn(x, s=self.s, dim=self.dim, norm=self.norm)
                r = fftn.real
                return r

        ref_net = None

        return (
            aten_fft_fftn(dim, s, norm),
            ref_net,
            ["aten::fft_fftn", "aten::real"],
        )

    @pytest.mark.parametrize("input_shape", [[64, 49], [64, 50], [64, 64]])
    @pytest.mark.parametrize("dim", [[0]])
    @pytest.mark.parametrize("s", [[49], [-1], [49], [5]])
    @pytest.mark.parametrize("norm", ["forward", "backward", "ortho", None])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_fftn(self, ie_device, precision, ir_version, input_shape, dim, s, norm):
        self.input_shape = input_shape
        # Unfrozen test would fail due to issues with prim::GetAttr containing lists, strings or none.
        self._test(*self.create_model(dim, s, norm), ie_device, precision, ir_version, custom_eps=1e-3,
                   freeze_model=True)
