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
