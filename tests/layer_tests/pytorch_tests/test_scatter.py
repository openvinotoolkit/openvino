# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestScatter(PytorchLayerTest):
    def _prepare_input(self, dtype, out=False):
        inp = np.random.randn(6, 6).astype(dtype)
        if not out:
            return (inp,)
        return (inp, np.zeros_like(inp, dtype=dtype))

    def create_model(self, dim, index, src, inplace, reduce, has_out):
        class aten_scatter(torch.nn.Module):
            def __init__(self, dim, index, src, inplace, reduce, has_out=False):
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
                    str_forward += ("_out_of_place" if not has_out else "_with_out")


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

            def _forward_with_out(self, x: torch.Tensor, out: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter(x, self.dim, index, self.src, out=out)

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

            def _forward_with_out_reduce(self, x: torch.Tensor, out:torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter(x, self.dim, index, self.src, reduce=self.reduce, out=out)

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

        return aten_scatter(dim, index, src, inplace, reduce, has_out), ref_net, op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
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
    @pytest.mark.parametrize(["inplace", "has_out"], [(True, False), (False, True), (False, False)])
    @pytest.mark.parametrize("reduce", [None, "add", "multiply"])
    def test_scatter(self, dim, index, src, dtype, inplace, has_out, reduce, ie_device, precision, ir_version):
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
            *self.create_model(dim, index, src, inplace, reduce, has_out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype, "out": has_out},
            freeze_model=freeze,
        )


class TestScatterReduce(PytorchLayerTest):
    def _prepare_input(self, dtype, out=False):
        inp = np.random.randn(6, 6).astype(dtype)
        if not out:
            return (inp,)
        return (inp, np.zeros_like(inp, dtype=dtype))

    def create_model(self, dim, index, src, inplace, reduce, include_self, has_out):
        class aten_scatter_reduce(torch.nn.Module):
            def __init__(self, dim, index, src, inplace, reduce, include_self, has_out=False):
                super(aten_scatter_reduce, self).__init__()
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
                    str_forward += ("_out_of_place" if not has_out else "_with_out")

                self.reduce = reduce
                self.include_self = include_self
                self.forward = getattr(self, str_forward)

            def _forward_out_of_place(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter_reduce(x, self.dim, index, self.src, self.reduce, include_self=self.include_self)

            def _forward_with_out(self, x: torch.Tensor, out: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return torch.scatter_reduce(x, self.dim, index, self.src, self.reduce, include_self=self.include_self, out=out)

            def _forward_inplace(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                return x.scatter_reduce_(self.dim, index, self.src, self.reduce, include_self=self.include_self)

        ref_net = None
        if inplace:
            op_name = "aten::scatter_reduce_"
        else:
            op_name = "aten::scatter_reduce"

        return aten_scatter_reduce(dim, index, src, inplace, reduce, include_self, has_out), ref_net, op_name

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
    @pytest.mark.parametrize("src", [torch.arange(1, 26).reshape(5, 5)])
    @pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize(["inplace", "has_out"], [(True, False), (False, True), (False, False)])
    @pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
    @pytest.mark.parametrize("include_self", [True, False])
    def test_scatter_reduce(self, dim, index, src, dtype, inplace, has_out, reduce,  include_self, ie_device, precision, ir_version):
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
        kwargs = dict(kwargs_to_prepare_input={"dtype": dtype, "out": has_out}, freeze_model=freeze)
        if reduce == "mean" and dtype in ["int32", "int64"]:
            # rounding can be different on torch vs ov
            kwargs["custom_eps"] = 1.
        self._test(
            *self.create_model(dim, index, src, inplace, reduce, include_self, has_out),
            ie_device,
            precision,
            ir_version,
            **kwargs
        )

class TestScatterAdd(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (np.random.randn(6, 6).astype(dtype),)

    def create_model(self, dim, index, src, inplace):
        class aten_scatter_reduce(torch.nn.Module):
            def __init__(self, dim, index, src, inplace):
                super(aten_scatter_reduce, self).__init__()
                self.dim = dim
                self.use_empty_index = False
                if index is None:
                    self.use_empty_index = True
                    # Placeholder
                    self.index = torch.empty([1])
                else:
                    self.index = index
                self.src = src
                self.inplace = inplace

            def forward(self, x: torch.Tensor):
                if self.use_empty_index:
                    index = torch.empty([0, 0])
                else:
                    index = self.index
                if self.inplace:
                    return x.scatter_add_(self.dim, index, self.src)
                else:
                    return x.scatter_add(self.dim, index, self.src)

        op_name = "aten::scatter_add_" if inplace else "aten::scatter_add"

        return aten_scatter_reduce(dim, index, src, inplace), None, op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dim", [1, -1, 0])
    @pytest.mark.parametrize(
        "index",
        [
            None,  # Empty tensor scenario.
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[0, 5], [4, 1], [2, 3]]),
        ],
    )
    @pytest.mark.parametrize("src", [torch.arange(1, 26).reshape(5, 5)])
    @pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    def test_scatter_add(self, dim, index, src, dtype, inplace, ie_device, precision, ir_version):
        if isinstance(src, torch.Tensor):
            src = src.to(getattr(torch, dtype))
        if index is None:
            pytest.skip(
                "Cannot test reduce parameters with empty indexes due to issues with empty constant tensor or issues with prim::GetAttr str inputs."
            )
        self._test(
            *self.create_model(dim, index, src, inplace),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype},
        )
