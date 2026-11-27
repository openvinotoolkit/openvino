# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestSize(PytorchLayerTest):
    def _prepare_input(self, input_shape, complex_type):
        if complex_type:
            input_shape += [2]
        return (self.random.randn(*input_shape),)

    def create_model(self, complex_type):
        import torch

        class aten_size(torch.nn.Module):
            def __init__(self, complex_type):
                super().__init__()
                self.complex_type = complex_type

            def forward(self, x):
                if self.complex_type:
                    x = torch.view_as_complex(x)
                return torch.tensor(x.shape)

        op = aten_size(complex_type)

        return op, "aten::size"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[1,],
                                             [1, 2],
                                             [1, 2, 3],
                                             [1, 2, 3, 4],
                                             [1, 2, 3, 4, 5]])
    @pytest.mark.parametrize("complex_type", [True, False])
    def test_size(self, input_shape, complex_type, ie_device, precision, ir_version):
        self._test(*self.create_model(complex_type), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape,
                                            "complex_type": complex_type})

class TestSymFloat(PytorchLayerTest):
    def _prepare_input(self, rows):
        return (self.random.randn(rows, 2),)

    @pytest.mark.parametrize("rows", [2, 4, 9])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_sym_float(self, rows, ie_device, precision, ir_version):
        if not self.use_torch_export():
            pytest.skip("torch.sym_float only appears in exported graphs")

        class Model(torch.nn.Module):
            def forward(self, x):
                return x * torch.sym_float(x.shape[0])

        self._test(Model(), "aten::mul", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"rows": rows},
                   dynamic_shapes_for_export={"x": {0: torch.export.Dim.AUTO}},
                   fx_kind="aten.mul")
        
