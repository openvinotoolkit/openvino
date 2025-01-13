# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestIndexAdd(PytorchLayerTest):
    def _prepare_input(self, dtype, out):
        if not out:
            return (np.ones((3, 3)).astype(dtype),)
        return (np.ones((3, 3)).astype(dtype), np.zeros((3, 3)).astype(dtype))

    def create_model(self, dim, index, src, mode, alpha):
        class aten_index_add(torch.nn.Module):
            def __init__(self, dim, index, src, mode, alpha):
                super(aten_index_add, self).__init__()
                self.dim = dim
                self.index = index
                self.src = src                    
                self.inplace = mode == "inplace"
                self.alpha = alpha
                if mode == "out":
                    self.forward = self.forward_out

            def forward(self, x: torch.Tensor):
                index = self.index
                if self.inplace:
                    return x.index_add_(self.dim, index, self.src, alpha=self.alpha), x
                else:
                    return torch.index_add(x, self.dim, index, self.src, alpha=self.alpha), x

            def forward_out(self, x: torch.Tensor, out):
                index = self.index
                return torch.index_add(x, self.dim, index, self.src, out=out, alpha=self.alpha), out

        op_name = "aten::index_add_" if mode == "inplace" else "aten::index_add"

        return aten_index_add(dim, index, src, mode, alpha), None, op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize(
        "index",
        [
            torch.tensor([0, 2, 1]),
            torch.tensor([0, 0, 0])
        ],
    )
    @pytest.mark.parametrize("src", [torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])])
    @pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("mode", ["inplace", "out", "default"])
    @pytest.mark.parametrize("alpha", [1, -1, 0.5, 0.25])
    def test_scatter_reduce(self, dim, index, src, dtype, mode, alpha, ie_device, precision, ir_version):
        if isinstance(src, torch.Tensor):
            src = src.to(getattr(torch, dtype))
        self._test(
            *self.create_model(dim, index, src, mode, alpha),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype, "out": mode == "out"},
        )
