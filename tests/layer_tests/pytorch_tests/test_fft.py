# Copyright (C) 2018-2025 Intel Corporation
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
        self._test(*self.create_model(dim, s, norm), ie_device, precision, ir_version, custom_eps=1e-3)


class aten_fft(torch.nn.Module):
    def __init__(self, op, n, dim, norm):
        super().__init__()
        self.n = n
        self.dim = dim
        self.norm = norm
        self.op = op

    def forward(self, x):
        if x.shape[-1] == 2:
            x = torch.view_as_complex(x)
        res = self.op(x, self.n, dim=self.dim, norm=self.norm)
        if res.dtype.is_complex:
            return torch.view_as_real(res)
        return res


class TestFFT(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.input_shape).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[67], [80], [12, 14], [9, 6, 3]])
    @pytest.mark.parametrize("n", [None, 50, 6])
    @pytest.mark.parametrize("dim", [-1, 0])
    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    @pytest.mark.parametrize("op,aten_name,in_complex", [
        (torch.fft.fft, "aten::fft_fft", True),
        (torch.fft.fft, "aten::fft_fft", False),
        pytest.param(torch.fft.hfft, "aten::fft_hfft", True, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.rfft, "aten::fft_rfft", False),
        (torch.fft.ifft, "aten::fft_ifft", True),
        pytest.param(torch.fft.ihfft, "aten::fft_ihfft", False, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.irfft, "aten::fft_irfft", True),
    ])
    def test_1d(self, ie_device, precision, ir_version, input_shape, op, n, dim, norm, aten_name, in_complex):
        if op in [torch.fft.rfft, torch.fft.irfft] and n is not None and input_shape[dim] < n:
            pytest.skip("Signal size greater than input size is not supported yet")
        if in_complex:
            self.input_shape = input_shape + [2]
        else:
            self.input_shape = input_shape
        m = aten_fft(op, n, dim, norm)
        self._test(m, None, aten_name, ie_device,
                   precision, ir_version, trace_model=True, dynamic_shapes=False, custom_eps=1e-3)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[20, 30], [15, 20, 30], [10, 15, 20, 30]])
    @pytest.mark.parametrize("s", [None, (10, 10)])
    @pytest.mark.parametrize("dim", [(0, 1), (-2, -1)])
    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    @pytest.mark.parametrize("op,aten_name,in_complex", [
        (torch.fft.fft2, "aten::fft_fft2", True),
        (torch.fft.fft2, "aten::fft_fft2", False),
        pytest.param(torch.fft.hfft2, "aten::fft_hfft2", True, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.rfft2, "aten::fft_rfft2", False),
        (torch.fft.ifft2, "aten::fft_ifft2", True),
        pytest.param(torch.fft.ihfft2, "aten::fft_ihfft2", False, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.irfft2, "aten::fft_irfft2", True),
    ])
    def test_2d(self, ie_device, precision, ir_version, input_shape, op, s, dim, norm, aten_name, in_complex):
        if in_complex:
            self.input_shape = input_shape + [2]
        else:
            self.input_shape = input_shape
        m = aten_fft(op, s, dim, norm)
        self._test(m, None, aten_name, ie_device,
                   precision, ir_version, trace_model=True, dynamic_shapes=False, custom_eps=1e-3)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape,s,dim", [
        ((4, 5), None, None),
        ((4, 5), None, (0,)),
        ((4, 5), None, (0, -1)),
        ((4, 5, 6), None, None),
        ((4, 5, 6), None, (0,)),
        ((4, 5, 6), None, (0, -1)),
        ((4, 5, 6, 7), None, None),
        ((4, 5, 6, 7), None, (0,)),
        ((4, 5, 6, 7), None, (0, -1)),
        ((4, 5, 6, 7, 8, 4), None, None),
        ((4, 5, 6, 7, 8), None, (1, 3, 4)),
        ((4, 5, 6), None, (1,)),
        ((4,), None, (0,)),
        ((4, 5, 60, 70), (10, 10), None),
        ((40, 50, 6, 7), (10, 10), (0, 1)),
    ])
    @pytest.mark.parametrize("norm", [None, "forward", "backward", "ortho"])
    @pytest.mark.parametrize("op,aten_name,in_complex", [
        (torch.fft.fftn, "aten::fft_fftn", True),
        (torch.fft.fftn, "aten::fft_fftn", False),
        pytest.param(torch.fft.hfftn, "aten::fft_hfftn", True, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.rfftn, "aten::fft_rfftn", False),
        (torch.fft.ifftn, "aten::fft_ifftn", True),
        pytest.param(torch.fft.ihfftn, "aten::fft_ihfftn", False, marks=pytest.mark.skip(reason="Not supported yet.")),
        (torch.fft.irfftn, "aten::fft_irfftn", True),
    ])
    def test_nd(self, ie_device, precision, ir_version, input_shape, op, s, dim, norm, aten_name, in_complex):
        if in_complex:
            self.input_shape = input_shape + (2,)
        else:
            self.input_shape = input_shape
        m = aten_fft(op, s, dim, norm)
        self._test(m, None, aten_name, ie_device,
                   precision, ir_version, trace_model=True, dynamic_shapes=False, custom_eps=1e-3)
