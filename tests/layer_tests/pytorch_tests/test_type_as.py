# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTypeAs(PytorchLayerTest):
    def _prepare_input(self, input_dtype=np.float32, cast_dtype=np.float32):
        input_data = self.random.randint(127, size=(1, 3, 224, 224))
        return (input_data.astype(input_dtype), input_data.astype(cast_dtype))

    def create_model(self):
        import torch

        class aten_type_as(torch.nn.Module):

            def forward(self, x, y):
                return x.type_as(y)


        return aten_type_as(), "aten::type_as"

    @pytest.mark.parametrize("input_dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint8])
    @pytest.mark.parametrize("cast_dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint8])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_type_as(self, input_dtype, cast_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_dtype": input_dtype, "cast_dtype": cast_dtype})


class TestComplexTypeAs(PytorchLayerTest):
    """Test aten::type_as with complex tensors."""

    def _prepare_input(self, input_dtype=np.float32, cast_dtype=np.float64):
        # Complex tensor represented as real with shape [..., 2]
        return (self.random.randn(2, 4, 2, dtype=input_dtype),
                self.random.randn(2, 4, 2, dtype=cast_dtype))

    def create_model(self):
        import torch

        class ComplexTypeAs(torch.nn.Module):
            def forward(self, x, y):
                # Convert to complex, use type_as, convert back
                cx = torch.view_as_complex(x)
                cy = torch.view_as_complex(y)
                result = cx.type_as(cy)
                return torch.view_as_real(result)

        return ComplexTypeAs(), "aten::type_as"

    @pytest.mark.parametrize("input_dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("cast_dtype", [np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_type_as(self, input_dtype, cast_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_dtype": input_dtype, "cast_dtype": cast_dtype},
                   trace_model=True)
