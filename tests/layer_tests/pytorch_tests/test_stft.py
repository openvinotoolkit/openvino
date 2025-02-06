# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSTFT(PytorchLayerTest):
    def _prepare_input(self, win_length, signal_shape, rand_data=False, out_dtype="float32"):
        import numpy as np

        if rand_data:
            signal = np.random.randn(*signal_shape).astype(out_dtype)
        else:
            num_samples = signal_shape[-1]
            half_idx = num_samples // 2
            t = np.linspace(0, 1, num_samples)
            signal = np.sin(2 * np.pi * 5 * t)
            signal[half_idx:] += np.sin(2 * np.pi * 10 * t[half_idx:])
            signal = np.broadcast_to(signal, signal_shape).astype(out_dtype)

        window = np.hanning(win_length).reshape([win_length])

        return (signal, window.astype(out_dtype))

    def create_model(self, n_fft, hop_length, win_length, normalized):
        import torch

        class aten_stft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, normalized):
                super(aten_stft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.normalized = normalized

            def forward(self, x, window):
                return torch.stft(
                    x,
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=False,
                    pad_mode="reflect",
                    normalized=self.normalized,
                    onesided=True,
                    return_complex=False,
                )

        ref_net = None

        return aten_stft(n_fft, hop_length, win_length, normalized), ref_net, "aten::stft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.parametrize(("signal_shape"), [(1, 256), (2, 128), (128,)])
    @pytest.mark.parametrize(("n_fft", "hop_length", "window_size"), [
        [16, 4, 16],
        [32, 32, 32],
        [32, 16, 24],
        [24, 32, 20],
        [128, 128, 128],
    ])
    @pytest.mark.parametrize(("normalized"), [True, False])
    def test_stft(self, n_fft, hop_length, window_size, signal_shape, normalized, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="STFT op is not supported on GPU yet")
        self._test(*self.create_model(n_fft, hop_length, window_size, normalized), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": window_size, "signal_shape": signal_shape}, trace_model=trace_model)


class TestSTFTAttrs(PytorchLayerTest):
    def _prepare_input(self, out=False, out_dtype="float32"):
        import numpy as np

        signal = np.random.randn(2, 512).astype(out_dtype)
        return (signal,)

    def create_model_with_attrs(self, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex):
        import torch

        class aten_stft_attrs(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex):
                super(aten_stft_attrs, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.window = None  # Default window
                self.center = center
                self.pad_mode = pad_mode
                self.normalized = normalized
                self.onesided = onesided
                self.return_complex = return_complex

            def forward(self, x):
                stft = torch.stft(
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
                if self.return_complex:
                    return torch.view_as_real(stft)
                else:
                    return stft

        ref_net = None

        return aten_stft_attrs(n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex), ref_net, "aten::stft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.parametrize(("n_fft", "hop_length", "win_length", "center", "pad_mode", "normalized", "onesided", "return_complex"), [
        [16, 4, 16, False, "reflect", False, True, False],  # default window
        [16, 4, 14, True, "reflect", False, True, False],  # center True
        [16, 4, 14, True, "reflect", False, True, False],  # center True
        [16, 4, 14, True, "replicate", False, True, False],  # center True
        [16, 4, 14, False, "replicate", False, True, False],  # center False
        [16, None, 16, False, "reflect", False, True, False],  # hop_length None
        [16, None, None, False, "reflect", False, True, False],  # hop & win length None
        [16, 4, None, False, "reflect", False, True, False],  # win_length None
        [16, 4, 16, False, "reflect", True, True, False],  # normalized True
        [16, 4, 16, False, "reflect", False, True, True],  # return_complex True
        # Unsupported cases:
        [16, 4, 16, False, "reflect", False, False, False],  # onesided False
    ])
    def test_stft_not_supported_attrs(self, n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="STFT op is not supported on GPU yet")

        if center is True and trace_model is False:
            pytest.xfail(
                reason="torch stft uses list() for `center` subgrpah before aten::stft, that leads to error: No conversion rule found for operations: aten::list")

        if onesided is False:
            pytest.xfail(
                reason="aten::stft conversion is currently supported with onesided=True only")

        self._test(*self.create_model_with_attrs(n_fft, hop_length, win_length, center, pad_mode, normalized, onesided, return_complex), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={}, trace_model=trace_model)
