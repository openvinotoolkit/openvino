# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestComplex(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2).astype(np.float32),)

    def create_model(self, dtype):
        class complex_model(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype
                x = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j], dtype=dtype)
                self.complex_attr = torch.nn.Parameter(x)

            def forward(self, x):
                complex_const = torch.tensor(
                    [5.0 + 6.0j, 7.0 + 8.0j], dtype=self.dtype)

                real_attr = self.complex_attr.real
                imag_attr = self.complex_attr.imag

                real_const = complex_const.real
                imag_const = complex_const.imag

                real_result = x + real_attr + real_const
                imag_result = imag_attr + imag_const

                result = real_result + imag_result
                return result

        return complex_model(dtype), None, ["prim::GetAttr", "prim::Constant"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [
        torch.complex32,
        torch.complex64,
        torch.complex128])
    def test_complex(self, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype), ie_device,
                   precision, ir_version, trace_model=True)


class TestReal(PytorchLayerTest):
    def _prepare_input(self, y):
        import numpy as np
        if y:
            return (np.random.randn(2, 3).astype(np.float32),
                    np.random.randn(2, 3).astype(np.float32))
        return (np.random.randn(2, 3).astype(np.float32),)

    def create_model(self):
        class aten_real(torch.nn.Module):
            def forward(self, x, y=None):
                if y is not None:
                    c = torch.complex(x, y)
                else:
                    c = x
                return torch.real(c)

        return aten_real(), None, "aten::real"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("second_input", [True, False])
    def test_complex(self, second_input, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"y": second_input})


class TestComplexOutput(PytorchLayerTest):
    """
    Test that models with complex tensor outputs can be converted.
    This tests the ComplexTypeMarkRemover transformation which removes
    ComplexTypeMark nodes from the graph before model finalization.
    """

    def _prepare_input(self):
        import numpy as np
        # Input shape [batch, features, 2] where 2 is for complex (real, imag)
        return (np.random.randn(2, 3, 2).astype(np.float32),)

    def create_model(self, op_name):
        class complex_output_unsqueeze(torch.nn.Module):
            def forward(self, x):
                # Convert real tensor to complex, apply unsqueeze, convert back
                complex_x = torch.view_as_complex(x)
                result = complex_x.unsqueeze(0)
                return torch.view_as_real(result)

        class complex_output_reshape(torch.nn.Module):
            def forward(self, x):
                # Convert real tensor to complex, apply reshape, convert back
                complex_x = torch.view_as_complex(x)
                result = complex_x.reshape(-1)
                return torch.view_as_real(result)

        class complex_output_permute(torch.nn.Module):
            def forward(self, x):
                # Convert real tensor to complex, apply permute, convert back
                complex_x = torch.view_as_complex(x)
                result = complex_x.permute(1, 0)
                return torch.view_as_real(result)

        models = {
            "unsqueeze": complex_output_unsqueeze,
            "reshape": complex_output_reshape,
            "permute": complex_output_permute,
        }

        return models[op_name](), None, f"aten::{op_name}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_name", ["unsqueeze", "reshape", "permute"])
    def test_complex_output(self, op_name, ie_device, precision, ir_version):
        self._test(*self.create_model(op_name), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexMulWithBuffer(PytorchLayerTest):
    """
    Test complex multiplication with buffer (CVS-176305).
    This tests that complex buffers from prim::GetAttr are correctly
    wrapped in ComplexTypeMark and preserved through operations.
    """

    def _prepare_input(self):
        import numpy as np
        # Input: [batch, seq, features] - will be reshaped to complex
        return (np.random.randn(2, 4, 16).astype(np.float32),)

    def create_model(self, freqs_dtype):
        class complex_mul_with_buffer(torch.nn.Module):
            def __init__(self, freqs_dtype):
                super().__init__()
                # Register complex buffer
                freqs = torch.randn(4, 8, 2, dtype=freqs_dtype)
                self.register_buffer('freqs', torch.view_as_complex(freqs))

            def forward(self, x):
                # Reshape input for complex view
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())
                # Multiply with complex buffer
                result = x_complex * self.freqs
                return torch.view_as_real(result).flatten(-2)

        return complex_mul_with_buffer(freqs_dtype), None, "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float64])
    def test_complex_mul_with_buffer(self, freqs_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(freqs_dtype), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexRoPEPattern(PytorchLayerTest):
    """
    Test Qwen-Image-like RoPE pattern with complex multiplication (CVS-176305).
    This tests the full RoPE pattern: view_as_complex -> unsqueeze -> mul -> view_as_real.
    """

    def _prepare_input(self):
        import numpy as np
        # Input: [batch, seq, heads, head_dim]
        return (np.random.randn(1, 8, 4, 16).astype(np.float32),)

    def create_model(self):
        class rope_pattern(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # freqs_cis: [seq, head_dim//2] complex
                freqs = torch.randn(8, 8, 2, dtype=torch.float32)
                self.register_buffer('freqs_cis', torch.view_as_complex(freqs))

            def forward(self, x):
                # Reshape for complex: [batch, seq, heads, head_dim//2, 2]
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())

                # Add dimensions for broadcasting: [1, seq, 1, head_dim//2]
                freqs = self.freqs_cis.unsqueeze(0).unsqueeze(2)

                # Complex multiplication with broadcasting
                x_rotated = x_complex * freqs

                return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

        return rope_pattern(), None, "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_rope_pattern(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexSqueeze(PytorchLayerTest):
    """
    Test squeeze preserves ComplexTypeMark (CVS-176305).
    """

    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 16).astype(np.float32),)

    def create_model(self, squeeze_dim):
        class complex_squeeze(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                freqs = torch.randn(1, 4, 8, 2)
                self.register_buffer('freqs', torch.view_as_complex(freqs))

            def forward(self, x):
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())
                freqs = self.freqs.squeeze(self.dim)
                result = x_complex * freqs
                return torch.view_as_real(result).flatten(-2)

        return complex_squeeze(squeeze_dim), None, "aten::squeeze"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("squeeze_dim", [0, -3])
    def test_complex_squeeze(self, squeeze_dim, ie_device, precision, ir_version):
        self._test(*self.create_model(squeeze_dim), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexSelect(PytorchLayerTest):
    """
    Test select preserves ComplexTypeMark (CVS-176305).
    """

    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 16).astype(np.float32),)

    def create_model(self):
        class complex_select(torch.nn.Module):
            def __init__(self):
                super().__init__()
                freqs = torch.randn(4, 4, 8, 2)
                self.register_buffer('freqs', torch.view_as_complex(freqs))

            def forward(self, x):
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())
                freqs = self.freqs.select(0, 0)
                result = x_complex * freqs
                return torch.view_as_real(result).flatten(-2)

        return complex_select(), None, "aten::select"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_select(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexNarrow(PytorchLayerTest):
    """
    Test narrow preserves ComplexTypeMark (CVS-176305).
    """

    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 4, 16).astype(np.float32),)

    def create_model(self):
        class complex_narrow(torch.nn.Module):
            def __init__(self):
                super().__init__()
                freqs = torch.randn(8, 8, 2)
                self.register_buffer('freqs', torch.view_as_complex(freqs))

            def forward(self, x):
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())
                freqs = self.freqs.narrow(0, 0, 4)
                result = x_complex * freqs
                return torch.view_as_real(result).flatten(-2)

        return complex_narrow(), None, "aten::narrow"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_narrow(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, trace_model=True)


class TestComplexTypeAsChain(PytorchLayerTest):
    """
    Test CVS-176305: type_as in complex operation chain.
    Tests the full RoPE pattern with type conversion.
    """

    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 8, 4, 16).astype(np.float32),)

    def create_model(self, freqs_dtype):
        class rope_with_type_as(torch.nn.Module):
            def __init__(self, freqs_dtype):
                super().__init__()
                freqs = torch.randn(8, 8, 2, dtype=freqs_dtype)
                self.register_buffer('freqs_cis', torch.view_as_complex(freqs))

            def forward(self, x):
                x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
                x_complex = torch.view_as_complex(x_reshaped.float())
                freqs = self.freqs_cis.unsqueeze(0).unsqueeze(2)
                freqs = freqs.type_as(x_complex)
                x_rotated = x_complex * freqs
                return torch.view_as_real(x_rotated).flatten(-2).type_as(x)

        return rope_with_type_as(freqs_dtype), None, "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("freqs_dtype", [torch.float32, torch.float64])
    def test_rope_with_type_as(self, freqs_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(freqs_dtype), ie_device,
                   precision, ir_version, trace_model=True)
