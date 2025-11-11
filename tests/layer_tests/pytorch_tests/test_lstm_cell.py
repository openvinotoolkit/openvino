
# import numpy as np
# import pytest
# import torch
# import torch.nn as nn
# from pytorch_layer_test_class import PytorchLayerTest


# class aten_lstm_cell(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, has_bias):
#         super().__init__()
#         self.cell = nn.LSTMCell(input_size, hidden_size, bias=has_bias)

#     # Forward signature must match what TorchScript expects:
#     # (x, (h0, c0))  -->  returns (h1, c1)
#     def forward(self, x, hc):
#         h0, c0 = hc
#         return self.cell(x, (h0, c0))


# class TestLSTMCell(PytorchLayerTest):

#     # ----------------------------------------------------------------------
#     # Utility to generate random tensors
#     # ----------------------------------------------------------------------
#     def _prepare_input(self, batch=3, input_size=10, hidden_size=20):
#         np.random.seed(42)
#         x = np.random.randn(batch, input_size).astype(np.float32)
#         h0 = np.random.randn(batch, hidden_size).astype(np.float32)
#         c0 = np.random.randn(batch, hidden_size).astype(np.float32)
#         # Important: return (x, (h0, c0)) to match the model’s forward() signature
#         return (x, (h0, c0))

#     # ----------------------------------------------------------------------
#     # 1. Multi-step forward "smoke" test (based on PyTorch test_LSTM_cell)
#     # ----------------------------------------------------------------------
#     @pytest.mark.parametrize("has_bias", [True, False])
#     def test_lstm_cell_multistep_forward(self, ie_device, precision, ir_version, has_bias):
#         if ie_device == "GPU":
#             pytest.skip("GPU LSTMCell not supported yet")

#         input_size, hidden_size = 10, 20
#         self.input_size, self.hidden_size = input_size, hidden_size
#         model = aten_lstm_cell(input_size, hidden_size, has_bias)

#         # Prepare initial tensors
#         x, (h, c) = self._prepare_input()
#         x, h, c = map(torch.tensor, (x, h, c))

#         # Run multiple steps (to check forward stability)
#         for _ in range(6):
#             h, c = model(x, (h.clone(), c.clone()))

#         # Now test conversion/export with correct example input structure
#         example_input = self._prepare_input()
#         self._test(
#             model,
#             None,
#             None,
#             ie_device,
#             precision,
#             ir_version,
#             trace_model=True,
#             example_input=example_input,
#         )

#     # ----------------------------------------------------------------------
#     # 2. Wrong input size should raise error (test_LSTM_cell_forward_input_size)
#     # ----------------------------------------------------------------------
#     def test_lstm_cell_bad_input_size(self, ie_device, precision, ir_version):
#         x = torch.randn(3, 11)
#         hx = torch.randn(3, 20)
#         cx = torch.randn(3, 20)
#         lstm = nn.LSTMCell(10, 20)
#         with pytest.raises(RuntimeError):
#             _ = lstm(x, (hx, cx))

#     # ----------------------------------------------------------------------
#     # 3. Wrong hidden size should raise error (test_LSTM_cell_forward_hidden_size)
#     # ----------------------------------------------------------------------
#     def test_lstm_cell_bad_hidden_size(self, ie_device, precision, ir_version):
#         x = torch.randn(3, 10)
#         hx = torch.randn(3, 21)
#         cx = torch.randn(3, 20)
#         lstm = nn.LSTMCell(10, 20)
#         with pytest.raises(RuntimeError):
#             _ = lstm(x, (hx, cx))
#         with pytest.raises(RuntimeError):
#             _ = lstm(x, (cx, hx))



# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# LSTMCell tests for OpenVINO PyTorch frontend
# Covers both core functional correctness and error handling
# Mirrors PyTorch test_nn.py behavior

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytorch_layer_test_class import PytorchLayerTest


class aten_lstm_cell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, has_bias):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size, bias=has_bias)

    def forward(self, x, h0, c0):
        return self.cell(x, (h0, c0))


class TestLSTMCell(PytorchLayerTest):

    def _prepare_input(self, batch=3, input_size=10, hidden_size=20):
        np.random.seed(42)
        x = np.random.randn(batch, input_size).astype(np.float32)
        h0 = np.random.randn(batch, hidden_size).astype(np.float32)
        c0 = np.random.randn(batch, hidden_size).astype(np.float32)
        # ✅ Wrap hidden and cell states into a tuple
        return (x, (h0, c0))

    # ----------------------------------------------------------------------
    # 1. Multi-step forward (smoke test)
    # ----------------------------------------------------------------------
    @pytest.mark.parametrize("has_bias", [True, False])
    def test_lstm_cell_multistep_forward(self, ie_device, precision, ir_version, has_bias):
        if ie_device == "GPU":
            pytest.skip("GPU LSTMCell not supported")
        input_size, hidden_size = 10, 20
        self.input_size, self.hidden_size = input_size, hidden_size
        model = nn.LSTMCell(input_size, hidden_size, bias=has_bias)
        x, (h, c) = self._prepare_input()
        for _ in range(6):
            h, c = model(torch.tensor(x), (torch.tensor(h), torch.tensor(c)))
        self._test(model, None, None, ie_device, precision, ir_version, trace_model=True)

    # ----------------------------------------------------------------------
    # 2. Bad input size
    # ----------------------------------------------------------------------
    def test_lstm_cell_bad_input_size(self, ie_device, precision, ir_version):
        input = torch.randn(3, 11)
        hx = torch.randn(3, 20)
        cx = torch.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        with pytest.raises(RuntimeError):
            _ = lstm(input, (hx, cx))

    # ----------------------------------------------------------------------
    # 3. Bad hidden size
    # ----------------------------------------------------------------------
    def test_lstm_cell_bad_hidden_size(self, ie_device, precision, ir_version):
        input = torch.randn(3, 10)
        hx = torch.randn(3, 21)
        cx = torch.randn(3, 20)
        lstm = nn.LSTMCell(10, 20)
        with pytest.raises(RuntimeError):
            _ = lstm(input, (hx, cx))
        with pytest.raises(RuntimeError):
            _ = lstm(input, (cx, hx))

    # ======================================================================
    # 🔹 ADDITIONAL COVERAGE TESTS (from PyTorch test_nn.py + extra)
    # ======================================================================

    # ----------------------------------------------------------------------
    # 4. Dtype consistency
    # ----------------------------------------------------------------------
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_lstm_cell_dtype_consistency(self, ie_device, precision, ir_version, dtype):
        if ie_device == "GPU":
            pytest.skip("Skip GPU for dtype consistency")
        model = nn.LSTMCell(10, 20).to(dtype)  # ✅ Match model weight dtype
        x = torch.randn(3, 10, dtype=dtype)
        h = torch.randn(3, 20, dtype=dtype)
        c = torch.randn(3, 20, dtype=dtype)
        y_h, y_c = model(x, (h, c))
        assert y_h.dtype == dtype
        assert y_c.dtype == dtype

    # ----------------------------------------------------------------------
    # 5. Zero input and hidden states
    # ----------------------------------------------------------------------
    def test_lstm_cell_zero_input(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("Skip GPU")
        model = nn.LSTMCell(5, 7)
        x = torch.zeros(1, 5)
        h = torch.zeros(1, 7)
        c = torch.zeros(1, 7)
        out_h, out_c = model(x, (h, c))
        assert out_h.shape == (1, 7)
        assert out_c.shape == (1, 7)
        assert torch.all(torch.isfinite(out_h))
        assert torch.all(torch.isfinite(out_c))

    # ----------------------------------------------------------------------
    # 6. Backward propagation (autograd smoke test)
    # ----------------------------------------------------------------------
    def test_lstm_cell_backward(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("Skip GPU")
        model = nn.LSTMCell(4, 6)
        x = torch.randn(2, 4, requires_grad=True)
        h = torch.randn(2, 6, requires_grad=True)
        c = torch.randn(2, 6, requires_grad=True)
        out_h, out_c = model(x, (h, c))
        loss = out_h.sum() + out_c.sum()
        loss.backward()
        assert x.grad is not None
        assert h.grad is not None
        assert c.grad is not None

    # ----------------------------------------------------------------------
    # 7. Random-length loop test (variable timesteps)
    # ----------------------------------------------------------------------
    def test_lstm_cell_random_steps(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("Skip GPU")
        torch.manual_seed(0)
        model = nn.LSTMCell(8, 12)
        x = torch.randn(1, 8)
        h = torch.randn(1, 12)
        c = torch.randn(1, 12)
        steps = np.random.randint(1, 10)
        for _ in range(steps):
            h, c = model(x, (h, c))
        assert h.shape == (1, 12)
        assert c.shape == (1, 12)

    # ----------------------------------------------------------------------
    # 8. Bias vs no-bias equivalence sanity check
    # ----------------------------------------------------------------------
    def test_lstm_cell_no_bias_equivalence(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("Skip GPU")
        m1 = nn.LSTMCell(5, 5, bias=True)
        m2 = nn.LSTMCell(5, 5, bias=False)
        with torch.no_grad():
            m2.weight_ih.copy_(m1.weight_ih)
            m2.weight_hh.copy_(m1.weight_hh)
        x = torch.randn(1, 5)
        h = torch.randn(1, 5)
        c = torch.randn(1, 5)
        out1 = m1(x, (h, c))
        out2 = m2(x, (h, c))
        # Bias affects outputs, so they should differ
        assert not torch.allclose(out1[0], out2[0])
