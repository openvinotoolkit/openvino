# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test Rotary Position Embedding (RoPE) pattern with complex tensors.
"""

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestRoPEPattern(PytorchLayerTest):
    """Test RoPE pattern: complex buffer -> unsqueeze -> view_as_real.

    This is the main pattern that caused issues in Qwen-Image model.
    """

    def _prepare_input(self, seq_len=8, dim=64):
        rng = np.random.default_rng(43)
        return (rng.standard_normal((1, seq_len, dim)).astype(np.float32),)

    def create_model(self, dim=64):
        class RoPEModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                # Create freqs_cis as complex buffer (like in Qwen-Image)
                g = torch.Generator().manual_seed(42)
                freqs = torch.randn(8, dim // 2, generator=g)
                self.register_buffer("freqs_cis",
                    torch.polar(torch.ones_like(freqs), freqs))

            def forward(self, x):
                batch, seq_len, dim = x.shape
                freqs = self.freqs_cis[:seq_len]
                freqs = freqs.unsqueeze(0)  # This was causing the bug
                freqs_real = torch.view_as_real(freqs)
                x_reshaped = x.view(batch, seq_len, dim // 2, 2)
                result = x_reshaped * freqs_real
                return result.view(batch, seq_len, dim)

        return RoPEModule(dim), None, ["aten::unsqueeze", "aten::view_as_real"]

    @pytest.mark.parametrize("dim", [32, 64, 128])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_rope_pattern(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dim": dim},
                   trace_model=True)


class TestComplexBufferUnsqueeze(PytorchLayerTest):
    """Test complex buffer + unsqueeze

    This is the minimal test case that reproduces the original bug:
    - Complex buffer (pos_embed) registered in module
    - Slicing the buffer
    - Unsqueeze operation
    - view_as_real to get output
    """

    def _prepare_input(self):
        rng = np.random.default_rng(43)
        return (rng.standard_normal((1, 4, 8)).astype(np.float32),)

    def create_model(self):
        class ComplexBufferUnsqueeze(torch.nn.Module):
            def __init__(self):
                super().__init__()
                g = torch.Generator().manual_seed(42)
                real = torch.randn(4, 8, generator=g)
                imag = torch.randn(4, 8, generator=g)
                self.register_buffer("pos_embed", torch.complex(real, imag))

            def forward(self, x):
                pos = self.pos_embed[:x.shape[1]]
                pos = pos.unsqueeze(0)  # Bug trigger
                pos_real = torch.view_as_real(pos)
                return x.unsqueeze(-1) + pos_real

        return ComplexBufferUnsqueeze(), None, "aten::unsqueeze"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_buffer_unsqueeze(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)


class TestComplexBufferMultipleOps(PytorchLayerTest):
    """Test complex buffer with multiple operations.

    Tests complex buffer going through multiple operations before view_as_real.
    """

    def _prepare_input(self):
        rng = np.random.default_rng(43)
        return (rng.standard_normal((1, 4, 2)).astype(np.float32),)

    def create_model(self):
        class ComplexBufferMultipleOps(torch.nn.Module):
            def __init__(self):
                super().__init__()
                g = torch.Generator().manual_seed(42)
                freqs = torch.randn(4, 2, generator=g)
                complex_freqs = torch.view_as_complex(freqs)
                self.register_buffer("freqs_a", complex_freqs)
                self.register_buffer("freqs_b", complex_freqs * 2)

            def forward(self, x):
                cx = torch.view_as_complex(x)
                # Multiple operations with complex buffers
                a = self.freqs_a.unsqueeze(0)
                b = self.freqs_b.unsqueeze(0)
                result = cx * a + cx * b
                return torch.view_as_real(result)

        return ComplexBufferMultipleOps(), None, ["aten::unsqueeze", "aten::mul"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_buffer_multiple_ops(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)
