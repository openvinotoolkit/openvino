# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestScatter(PytorchLayerTest):
    def _prepare_input(self, dtype):
        inp = np.random.randn(6, 6).astype(getattr(np, dtype))
        return (inp,)

    def create_model(self, dim, index, src, inplace, reduce):
        class aten_scatter(torch.nn.Module):
            def __init__(self, dim, index, src, inplace, reduce):
                super(aten_scatter, self).__init__()
                self.dim = dim
                self.use_empty_index = False
                if index is None:
                    self.use_empty_index = True
                    # Placeholder
                    self.index = torch.empty([1])
                else:
                    self.index = index
                self.src = src
                str_forward = "_forward"
                if inplace:
                    str_forward += "_inplace"
                else:
                    str_forward += "_out_of_place"

                if reduce:
                    self.reduce = reduce
                    str_forward += "_reduce"
                self.forward = getattr(self, str_forward)

            def _forward_out_of_place(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter(x, self.dim, index, self.src)

            def _forward_inplace(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return x.scatter_(self.dim, index, self.src)

            def _forward_out_of_place_reduce(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter(x, self.dim, index, self.src, reduce=self.reduce)

            def _forward_inplace_reduce(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return x.scatter_(self.dim, index, self.src, reduce=self.reduce)

        ref_net = None
        if inplace:
            op_name = "aten::scatter_"
        else:
            op_name = "aten::scatter"

        return aten_scatter(dim, index, src, inplace, reduce), ref_net, op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dim", [1, -1, 0])
    @pytest.mark.parametrize(
        "index",
        [
            None,  # Empty tensor scenario.
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[0, 5], [4, 1], [2, 3]]),
        ],
    )
    @pytest.mark.parametrize("src", [torch.arange(1, 26).reshape(5, 5), 1])
    @pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.parametrize("reduce", [None, "add", "multiply"])
    def test_scatter(self, dim, index, src, dtype, inplace, reduce, ie_device, precision, ir_version):
        if isinstance(src, torch.Tensor):
            src = src.to(getattr(torch, dtype))
        freeze = True
        if index is None:
            # Freeze creates empty constant tensor which isn't supported by OV.
            freeze = False
        if (not freeze) and reduce:
            pytest.skip(
                "Cannot test reduce parameters with empty indexes due to issues with empty constant tensor or issues with prim::GetAttr str inputs."
            )
        self._test(
            *self.create_model(dim, index, src, inplace, reduce),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype},
            freeze_model=freeze
        )
