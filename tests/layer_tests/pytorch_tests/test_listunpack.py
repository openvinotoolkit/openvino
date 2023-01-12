# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List
import pytest
import numpy as np
import torch


from pytorch_layer_test_class import PytorchLayerTest


class TestListUnpack(PytorchLayerTest):
    def _prepare_input(self):
        return (
            np.random.randn(8, 3, 512, 512),
            np.random.randn(1, 3, 224, 224),
            np.random.randn(10, 1, 8, 8),
            np.random.randn(1, 1, 1, 1),
        )

    def create_model_size_listunpack(self):
        class prim_listunpack(torch.nn.Module):
            def forward(self, in1, in2, in3, in4):
                a, b, c, d = in1.size()
                return a, b, c, d

        ref_net = None

        return (
            prim_listunpack(),
            ref_net,
            "prim::ListUnpack",
        )

    def create_model_size_slice_listunpack(self, slices):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, slices):
                self.start = slices[0]
                self.stop = slices[1]
                self.step = slices[2]
                super(prim_listunpack, self).__init__()

            def forward(self, in1, in2, in3, in4):
                a, b = in1.size()[self.start : self.stop : self.step]
                return a, b

        ref_net = None

        return prim_listunpack(slices), ref_net, "prim::ListUnpack"

    def create_model_listconstruct_append_listunpack(self):
        class prim_listunpack(torch.nn.Module):
            def forward(self, in1, in2, in3, in4):
                in_list = [in1, in2]
                in_list.append(in3)
                in_list.append(in4)
                a, b, c, d = in_list
                return a, b, c, d

        ref_net = None

        return prim_listunpack(), ref_net, "prim::ListUnpack"

    def create_model_listconstruct_getitem_listunpack(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super(prim_listunpack, self).__init__()

            def forward(self, in1, in2, in3, in4):
                items: List[List[int]] = []
                items.append(in1.size())
                items.append(in2.size())
                items.append(in3.size())
                items.append(in4.size())
                getitem_0 = items[self.idx]
                a, b, c, d = getitem_0
                return a, b, c, d

        ref_net = None

        return prim_listunpack(idx), ref_net, "prim::ListUnpack"

    @pytest.mark.nightly
    def test_size_listunpack(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model_size_listunpack(), ie_device, precision, ir_version
        )

    @pytest.mark.parametrize(
        "slices", [(0, 2, 1), (0, 4, 2), (-1, -3, -1), (-3, -1, 1)]
    )
    @pytest.mark.nightly
    def test_size_slice_listunpack(self, slices, ie_device, precision, ir_version):
        self._test(
            *self.create_model_size_slice_listunpack(slices),
            ie_device,
            precision,
            ir_version
        )

    @pytest.mark.nightly
    def test_listconstruct_append_listunpack(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model_listconstruct_append_listunpack(),
            ie_device,
            precision,
            ir_version
        )

    @pytest.mark.parametrize("idx", [-4, -3, -2, -1, 0, 1, 2, 3])
    @pytest.mark.nightly
    def test_listconstruct_getitem_listunpack(
        self, idx, ie_device, precision, ir_version
    ):
        self._test(
            *self.create_model_listconstruct_getitem_listunpack(idx),
            ie_device,
            precision,
            ir_version
        )
