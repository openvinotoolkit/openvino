# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestISTFT(PytorchLayerTest):
    def _prepare_input(self, n_fft, hop_length, win_length, center, normalized, signal_shape, signal_length = None, rand_data=False, out_dtype="float32"):
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
        signal = torch.from_numpy(signal)
        
        if win_length is None:
            window_size = n_fft
        else:
            window_size = win_length

        if center and hop_length and hop_length != n_fft:
            window = np.hanning(window_size)
        else: # If 'center' is false, the window can't contain zeros at the beginning
            window = np.hamming(window_size)

        window = window.astype(out_dtype)
        window_tensor = torch.from_numpy(window)

        stft_out = torch.stft(
                    signal,
                    n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window_tensor,
                    center=center,
                    pad_mode="reflect",
                    normalized=normalized,
                    onesided=True,
                    return_complex=True,
                )
        
        if (signal_length is not None):
            signal_length_input = np.array(signal_length)
            return (torch.view_as_real(stft_out).numpy().astype(out_dtype), window, signal_length_input)
        else:
            return (torch.view_as_real(stft_out).numpy().astype(out_dtype), window)

    def create_model(self, n_fft, hop_length, win_length, normalized, center):

        class aten_istft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, normalized, center):
                super(aten_istft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.normalized = normalized
                self.center = center


            def forward(self, x, window):
                return torch.istft(
                    torch.view_as_complex(x),
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=self.center,
                    normalized=self.normalized,
                    onesided=True,
                    return_complex=False,
                    length = None
                )

        ref_net = None

        return aten_istft(n_fft, hop_length, win_length, normalized, center), ref_net, "aten::istft"

    def create_model_with_sig_len(self, n_fft, hop_length, win_length, normalized, center):

        class aten_istft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, normalized, center):
                super(aten_istft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.normalized = normalized
                self.center = center


            def forward(self, x, window, sig_length):
                return torch.istft(
                    torch.view_as_complex(x),
                    self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=window,
                    center=self.center,
                    normalized=self.normalized,
                    onesided=True,
                    return_complex=False,
                    length = sig_length.item()
                )

        ref_net = None

        return aten_istft(n_fft, hop_length, win_length, normalized, center), ref_net, "aten::istft"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [False, True])
    @pytest.mark.parametrize(("signal_shape", "n_fft", "hop_length", "window_size"), [
        [(1, 48), 16, 16, 16],
        [(1, 48), 16, 8, 16],
        [(1, 48), 16, 4, 16],
        [(1, 48), 16, 1, 16],
        [(1, 48), 16, 4, 16],
        [(2, 48), 16, 4, 16],
        [(3, 48), 16, 4, 16],
        [(4, 48), 16, 4, 16],
        [(1, 256), 32, 16, 32],
        [(2, 256), 32, 16, 32],
        [(1, 256), 24, 16, 20],
        [(1, 256), 128, 128, 128],
        [(1, 256), 256, 256, 256],
        [(1, 47), 17, 5, 17],
        [(1, 47), 17, 5, 13],
        [(1, 256), 133, 77, 133],
        [(1, 48), 16, None, 16],
        [(1, 48), 16, 8, None],
        [(1, 48), 16, None, None],
    ])
    @pytest.mark.parametrize(("normalized"), [True, False])
    @pytest.mark.parametrize(("center"), [True, False])
    def test_istft(self, n_fft, hop_length, window_size, signal_shape, normalized, center, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="ISTFT op is not supported on GPU yet")

        if center is False and window_size and window_size < n_fft:
            pytest.skip(
                reason="torch istft doesn't allow for zeros padding in window, when `center` is false ")

        self._test(*self.create_model(n_fft, hop_length, window_size, normalized, center), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": window_size, "signal_shape": signal_shape, "n_fft": n_fft, "hop_length" : hop_length, "center": center, "normalized": normalized}, trace_model=trace_model)


    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [False, True])
    @pytest.mark.parametrize(("signal_shape", "n_fft", "hop_length", "window_size", "signal_length"), [
        [(3, 48), 16, 8, 16, 48],
        [(3, 48), 16, 8, 16, 32],
        [(3, 48), 16, 8, 16, 55],
    ])
    @pytest.mark.parametrize(("normalized"), [True, False])
    @pytest.mark.parametrize(("center"), [True, False])
    def test_istft_with_sig_len(self, n_fft, hop_length, window_size, signal_shape, normalized, center, signal_length, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="ISTFT op is not supported on GPU yet")

        self._test(*self.create_model_with_sig_len(n_fft, hop_length, window_size, normalized, center), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": window_size, "signal_shape": signal_shape, "n_fft": n_fft, "hop_length" : hop_length, "center": center, "normalized": normalized, "signal_length": signal_length}, trace_model=trace_model)


class TestISTFTDefaultParams(PytorchLayerTest):
    def _prepare_input(self, n_fft, hop_length, win_length, center, normalized, signal_shape, out_dtype="float32"):
        import numpy as np

        signal = np.random.randn(*signal_shape).astype(out_dtype)
        signal = torch.from_numpy(signal)

        stft_kwargs = {
            'n_fft': n_fft,
            'return_complex': True
        }

        if hop_length is not None:
            stft_kwargs['hop_length'] = hop_length
        if win_length is not None:
            stft_kwargs['win_length'] = win_length
        if center is not None:
            stft_kwargs['center'] = center
        if normalized is not None:
            stft_kwargs['normalized'] = normalized

        stft_out = torch.stft(
                    signal,
                    **stft_kwargs
                )
        
        return (torch.view_as_real(stft_out).numpy().astype(out_dtype))

    def create_model(self, n_fft, hop_length, win_length, normalized, center):

        class aten_istft(torch.nn.Module):

            def __init__(self, n_fft, hop_length, win_length, normalized, center):
                super(aten_istft, self).__init__()
                self.n_fft = n_fft
                self.hop_length = hop_length
                self.win_length = win_length
                self.normalized = normalized
                self.center = center

            def forward(self, x):
                istft_kwargs = {
                    'n_fft': self.n_fft,
                    'onesided': True,
                    'return_complex': False,
                }

                if self.hop_length is not None:
                    istft_kwargs['hop_length'] = self.hop_length
                if self.win_length is not None:
                    istft_kwargs['win_length'] = self.win_length
                if self.center is not None:
                    istft_kwargs['center'] = self.center
                if self.normalized is not None:
                    istft_kwargs['normalized'] = self.normalized

                return torch.istft(torch.view_as_complex(x), **istft_kwargs)

        ref_net = None

        return aten_istft(n_fft, hop_length, win_length, normalized, center), ref_net, "aten::istft"

   
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("trace_model"), [True])
    @pytest.mark.parametrize(("signal_shape", "n_fft", "hop_length", "window_size"), [
        [(1, 48), 16, None, 16],
        [(1, 48), 16, 8, None],
        [(1, 48), 16, None, None],
    ])
    @pytest.mark.parametrize(("normalized"), [True, None])
    @pytest.mark.parametrize(("center"), [False, None])
    def test_istft(self, n_fft, hop_length, window_size, signal_shape, normalized, center, ie_device, precision, ir_version, trace_model):
        if ie_device == "GPU":
            pytest.xfail(reason="ISTFT op is not supported on GPU yet")

        self._test(*self.create_model(n_fft, hop_length, window_size, normalized, center), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"win_length": window_size, "signal_shape": signal_shape, "n_fft": n_fft, "hop_length" : hop_length, "center": center, "normalized": normalized}, trace_model=trace_model)
