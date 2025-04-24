# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestGetItem(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, idx, case="size_with_getitem"):
        class aten_size_get_item(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                return x.shape[self.idx]

        class aten_size_get_item_with_if(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx: int = idx

            def forward(self, x):
                if x.shape[self.idx] > self.idx:
                    res = x.shape[self.idx]
                else:
                    res = x.shape[-self.idx]
                return res

        ref_net = None
        op_cls = {
            "getitem": (aten_size_get_item, ["aten::size", "aten::__getitem__"]),
            "getitem_with_if": (aten_size_get_item_with_if, ["aten::size", "aten::__getitem__", "prim::If"])
        }
        op, op_in_graph = op_cls[case]

        return op(idx), ref_net, op_in_graph

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ([1,], 0),
        ([1,], -1),
        ([1, 2], 0),
        ([1, 2], 1),
        ([1, 2], -1),
        ([1, 2], -2),
        ([1, 2, 3], 0),
        ([1, 2, 3], 1),
        ([1, 2, 3], 2),
        ([1, 2, 3], -1),
        ([1, 2, 3], -2),
        ([1, 2, 3], -3),
        ([1, 2, 3, 4], 0),
        ([1, 2, 3, 4], 1),
        ([1, 2, 3, 4], 2),
        ([1, 2, 3, 4], 3),
        ([1, 2, 3, 4], -1),
        ([1, 2, 3, 4], -2),
        ([1, 2, 3, 4], -3),
        ([1, 2, 3, 4], -4),
        ([1, 2, 3, 4, 5], 0),
        ([1, 2, 3, 4, 5], 1),
        ([1, 2, 3, 4, 5], 2),
        ([1, 2, 3, 4, 5], 3),
        ([1, 2, 3, 4, 5], 4),
        ([1, 2, 3, 4, 5], -1),
        ([1, 2, 3, 4, 5], -2),
        ([1, 2, 3, 4, 5], -3),
        ([1, 2, 3, 4, 5], -4),
        ([1, 2, 3, 4, 5], -5)])
    @pytest.mark.parametrize("case", ["getitem", "getitem_with_if"])
    def test_getitem(self, input_shape, idx, case, ie_device, precision, ir_version):
        self._test(*self.create_model(idx, case), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})


class aten_add_getitem(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        list = [x, 2*x]
        list2 = list + [3*x, 4*x]
        return list2[self.idx]


class TestAddGetItem(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 1, 3),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("idx", [-4, -3, -2, -1, 0, 1, 2, 3])
    def test_add_cat(self, ie_device, precision, ir_version, idx):
        self._test(aten_add_getitem(idx), None, ["aten::__getitem__", "aten::add", "prim::ListConstruct"],
                   ie_device, precision, ir_version, freeze_model=False, use_convert_model=True)
