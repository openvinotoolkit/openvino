# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestRFFTN(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(*self.input_shape),)

    def create_model(self, dim, s, norm):
        class aten_fft_rfftn(torch.nn.Module):
            def __init__(self, dim, s, norm):
                super().__init__()
                self.dim = dim
                self.s = s
                self.norm = norm

            def forward(self, x):
                rfftn = torch.fft.rfftn(x, s=self.s, dim=self.dim, norm=self.norm)
                r = rfftn.real
                i = rfftn.imag
                irfftn = torch.fft.irfftn(torch.complex(r, i), s=self.s, dim=self.dim, norm=self.norm)
                return irfftn, r, i


        return (
            aten_fft_rfftn(dim, s, norm),
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
    def __init__(self, op, n, dim, norm, in_complex=False):
        super().__init__()
        self.n = n
        self.dim = dim
        self.norm = norm
        self.op = op
        if in_complex:
            self.forward = self.forward_complex

    def forward(self, x):
        res = self.op(x, self.n, dim=self.dim, norm=self.norm)
        if res.dtype.is_complex:
            return torch.view_as_real(res)
        return res

    def forward_complex(self, x):
        x = torch.view_as_complex(x)
        res = self.op(x, self.n, dim=self.dim, norm=self.norm)
        if res.dtype.is_complex:
            return torch.view_as_real(res)
        return res


class TestFFT(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(*self.input_shape),)

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
        m = aten_fft(op, n, dim, norm, in_complex)
        self._test(m, aten_name, ie_device,
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
        m = aten_fft(op, s, dim, norm, in_complex)
        self._test(m, aten_name, ie_device,
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
        m = aten_fft(op, s, dim, norm, in_complex)
        self._test(m, aten_name, ie_device,
                   precision, ir_version, trace_model=True, dynamic_shapes=False, custom_eps=1e-3)


class aten_rfft_irfft(torch.nn.Module):
    """Real -> rfft -> irfft -> real round-trip.

    With ``run_decompositions_for_export=True`` the exported graph contains
    ``aten._fft_r2c`` and ``aten._fft_c2r`` (the lowered ops) instead of the
    higher-level ``aten.fft_rfft`` / ``aten.fft_irfft`` wrappers.
    """

    def __init__(self, n, dim, norm):
        super().__init__()
        self.n = n
        self.dim = dim
        self.norm = norm

    def forward(self, x):
        spec = torch.fft.rfft(x, n=self.n, dim=self.dim, norm=self.norm)
        return torch.fft.irfft(spec, n=self.n, dim=self.dim, norm=self.norm)


class aten_fft_ifft_real(torch.nn.Module):
    """Real ([..., 2]) -> view_as_complex -> fft -> ifft -> view_as_real.

    Exercises ``aten._fft_c2c`` in both forward and inverse directions when
    decomposition is enabled.
    """

    def __init__(self, dim, norm):
        super().__init__()
        self.dim = dim
        self.norm = norm

    def forward(self, x):
        c = torch.view_as_complex(x)
        c = torch.fft.fft(c, dim=self.dim, norm=self.norm)
        c = torch.fft.ifft(c, dim=self.dim, norm=self.norm)
        return torch.view_as_real(c)


@pytest.mark.skipif(not PytorchLayerTest.use_torch_export(),
                    reason="Lowered aten._fft_* ops are produced only by torch.export's "
                           "default decomposition pipeline.")
class TestLoweredFFT(PytorchLayerTest):
    """Tests for the lowered aten._fft_* ops produced by torch.export decomposition.

    The high-level ``aten.fft_*`` ops are kept by the OpenVINO frontend's
    custom decomposition table - so to actually drive
    ``aten._fft_c2r`` / ``_fft_r2c`` / ``_fft_c2c`` translators we run
    ``run_decompositions()`` after export. The model itself is written with
    the user-facing ``torch.fft`` API so the reference path stays valid.
    """

    def _prepare_input(self):
        return (self.random.randn(*self.input_shape).astype(np.float32),)

    @pytest.mark.parametrize("input_shape", [[16], [8, 16], [4, 8, 16]])
    @pytest.mark.parametrize("dim", [-1])
    # CPU plugin currently crashes on the Multiply/Divide that "forward" and
    # "ortho" emit on top of the (I)RDFT output. The same crash happens for
    # the high-level fft_irfft translator and is pre-existing - covered here
    # only with norms that bypass the post-norm scaling.
    @pytest.mark.parametrize("norm", [None, "backward"])
    @pytest.mark.parametrize("n", [None])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_fft_r2c_c2r(self, ie_device, precision, ir_version, input_shape, dim, norm, n):
        self.input_shape = input_shape
        m = aten_rfft_irfft(n, dim, norm)
        # The exported+decomposed graph holds both the forward (r2c) and the
        # inverse (c2r) lowered ops; assert both are present.
        self._test(m, ("aten::_fft_r2c", "aten::_fft_c2r"), ie_device, precision, ir_version,
                   fx_kind=("aten._fft_r2c", "aten._fft_c2r"),
                   run_decompositions_for_export=True,
                   custom_eps=1e-3)

    @pytest.mark.parametrize("input_shape", [[16, 2], [8, 16, 2], [4, 8, 16, 2]])
    @pytest.mark.parametrize("dim", [-1])
    # See note on test_fft_r2c_c2r about "forward" / "ortho".
    @pytest.mark.parametrize("norm", [None, "backward"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_fft_c2c(self, ie_device, precision, ir_version, input_shape, dim, norm):
        self.input_shape = input_shape
        m = aten_fft_ifft_real(dim, norm)
        self._test(m, "aten::_fft_c2c", ie_device, precision, ir_version,
                   fx_kind="aten._fft_c2c",
                   run_decompositions_for_export=True,
                   custom_eps=1e-3)
