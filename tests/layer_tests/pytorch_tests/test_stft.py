# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSTFT(PytorchLayerTest):
    def _prepare_input(self, win_length, out=False, out_dtype="float32"):
        import numpy as np

        signal = np.random.randn(1, 256).astype(out_dtype)
        window = np.hanning(win_length).reshape([win_length])

        return (np.array(signal, dtype=out_dtype), window.astype(out_dtype))

    def create_model(self, n_fft, hop_length, win_length):
        import torch

        class aten_stft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length):
                super(aten_stft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length

            def forward(self, x, window):
                return torch.stft(
                    x,
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=False,
                    pad_mode="reflect",
                    normalized=False,
                    onesided=True,
                    return_complex=False,
                )

        ref_net = None

        return aten_stft(n_fft, hop_length, win_length), ref_net, "aten::stft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.parametrize(("n_fft", "hop_length", "window_size"), [
        [16, 4, 16],
        [32, 32, 32],
        [32, 16, 24],
        [24, 32, 20],
        [256, 256, 256],
    ])
    def test_stft(self, n_fft, hop_length, window_size, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="STFT op is not supported on GPU yet")
        self._test(*self.create_model(n_fft, hop_length, window_size), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": window_size}, trace_model=trace_model)


class TestSTFTAttrs(PytorchLayerTest):
    def _prepare_input(self, win_length, default_window=False, out=False, out_dtype="float32"):
        import numpy as np

        signal = np.random.randn(2, 512).astype(out_dtype)
        return [np.array(signal, dtype=out_dtype), ]

    def create_model_with_attrs(self, n_fft, hop_length, win_length, default_window, center, pad_mode, normalized, onesided, return_complex):
        import torch

        class aten_stft_attrs(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, default_window, center, pad_mode, normalized, onesided, return_complex):
                super(aten_stft_attrs, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.window = default_window
                self.center = center
                self.pad_mode = pad_mode
                self.normalized = normalized
                self.onesided = onesided
                self.return_complex = return_complex

            def forward(self, x):
                return torch.stft(
                    x,
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode=self.pad_mode,
                    normalized=self.normalized,
                    onesided=self.onesided,
                    return_complex=self.return_complex,
                )

        ref_net = None

        return aten_stft_attrs(n_fft, hop_length, win_length, default_window, center, pad_mode, normalized, onesided, return_complex), ref_net, "aten::stft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.parametrize(("n_fft", "hop_length", "win_length", "default_window", "center", "pad_mode", "normalized", "onesided", "return_complex"), [
        [16, 4, 16, None, False, "reflect", False, True, False],  # default window
        [16, 4, 14, None, True, "reflect", False, True, False],  # center True
        [16, 4, 14, None, True, "reflect", False, True, False],  # center True
        [16, 4, 14, None, True, "replicate", False, True, False],  # center True
        [16, 4, 14, None, False, "replicate", False, True, False],  # center False
        [16, None, 16, None, False, "reflect", False, True, False], # hop_length None
        [16, None, None, None, False, "reflect", False, True, False], # hop_length & win_length None
        [16, 4, None, None, False, "reflect", False, True, False], # win_length None
        ## Unsupported cases:
        [16, 4, 16, None, False, "reflect", True, True, False],  # normalized True
        [16, 4, 16, None, False, "reflect", False, False, False],  # onesided False
        [16, 4, 16, None, False, "reflect", False, True, True],  # reutrn_complex True
    ])
    def test_stft_not_supported_attrs(self, n_fft, hop_length, win_length, default_window, center, pad_mode, normalized, onesided, return_complex, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="STFT op is not supported on GPU yet")

        if center is True and trace_model is False:
            pytest.xfail(
                reason="torch stft uses list() for `center` subgrpah before aten::stft, that leads to error: No conversion rule found for operations: aten::list")

        if normalized is True:
            pytest.xfail(
                reason="aten::stft conversion is currently supported with normalized=False only")

        if onesided is False:
            pytest.xfail(
                reason="aten::stft conversion is currently supported with onesided=True only")

        if return_complex is True:
            pytest.xfail(
                reason="aten::stft conversion is currently supported with return_complex=False only")

        self._test(*self.create_model_with_attrs(n_fft, hop_length, win_length, default_window, center, pad_mode, normalized, onesided, return_complex), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": win_length, "default_window": default_window}, trace_model=trace_model)
