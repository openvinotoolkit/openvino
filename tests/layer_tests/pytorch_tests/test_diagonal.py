# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDiagonal(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        return (self.random.randn(*shape, dtype=dtype),)

    def create_model(self, offset, dim1, dim2):
        class aten_diagonal(torch.nn.Module):
            def __init__(self, offset, dim1, dim2):
                super().__init__()
                self.offset = offset
                self.dim1 = dim1
                self.dim2 = dim2

            def forward(self, x):
                return torch.diagonal(x, offset=self.offset, dim1=self.dim1, dim2=self.dim2)

        return aten_diagonal(offset, dim1, dim2), "aten::diagonal"

    @pytest.mark.parametrize("shape,dim1,dim2", [
        ((5, 5), 0, 1),
        ((3, 4, 6), 1, 2),
        ((3, 4, 6), -1, -2),
        ((2, 3, 4, 5), 0, 2),
        ((2, 3, 4, 5), 3, 1),
        ((2, 4, 6, 4, 6), 1, 3),
        ((2, 6, 6, 4, 9), 1, 2),
        # empty input
        ((2, 0, 4), 1, 2),
    ])
    @pytest.mark.parametrize("offset", [0, 1, -1, 3, -3])
    @pytest.mark.parametrize("dtype", ["float32", "int64", "bool"])
    @pytest.mark.parametrize("dynamic_shapes", [True, False])
    @pytest.mark.parametrize("trace_model", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_diagonal(self, shape, dim1, dim2, offset, dtype, dynamic_shapes, trace_model,
                      ie_device, precision, ir_version):
        self._test(*self.create_model(offset, dim1, dim2), ie_device, precision, ir_version,
                   dynamic_shapes=dynamic_shapes, trace_model=trace_model,
                   dynamic_shapes_for_export={"x": tuple(torch.export.Dim.AUTO for _ in shape)}
                   if dynamic_shapes else {},
                   kwargs_to_prepare_input={"shape": shape, "dtype": dtype})
