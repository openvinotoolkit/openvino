# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSplit(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 10, 224, 224).astype(np.float32),)

    def create_model_split_getitem(self):
        class aten_split(torch.nn.Module):
            def __init__(self, split, axis, getitem):
                self.split = split
                self.axis = axis
                self.getitem = getitem
                super(aten_split, self).__init__()

            def forward(self, input):
                return torch.split(input, self.split, self.axis)[self.getitem]

        ref_net = None

        return (
            aten_split(self.split_param, self.axis, self.getitem),
            ref_net,
            "aten::split",
        )

    def create_model_split_listunpack(self):
        class aten_split(torch.nn.Module):
            def __init__(self, split, axis):
                self.split = split
                self.axis = axis
                super(aten_split, self).__init__()

            def forward(self, input):
                # Hardcode to test with ListUnpack
                a, b, c, d, e = torch.split(input, self.split, self.axis)
                return b

        ref_net = None

        return aten_split(self.split_param, self.axis), ref_net, "aten::split"

    # Test case - (split_param, axis), always split into 5 due to hardcoded number of outputs in ListUnpack test.
    test_cases = [
        (2, 1),
        (45, 2),
        (45, -1),
        ([2, 2, 2, 2, 2], 1),
        ([200, 20, 1, 1, 2], 2),
        ([20, 200, 1, 1, 2], -1),
    ]

    @pytest.mark.parametrize("params", test_cases)
    @pytest.mark.parametrize("getitem", [-5, -2, -1, 0, 1, 4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_split_getitem(self, params, getitem, ie_device, precision, ir_version):
        (self.split_param, self.axis) = params
        self.getitem = getitem
        self._test(*self.create_model_split_getitem(),
                   ie_device, precision, ir_version)

    @pytest.mark.parametrize("params", test_cases)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_split_listunpack(self, params, ie_device, precision, ir_version):
        (self.split_param, self.axis) = params
        self._test(
            *self.create_model_split_listunpack(), ie_device, precision, ir_version
        )


class TestSplitWithSizes(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(20).astype(np.float32),np.random.randn(20).astype(np.float32))

    def create_model(self):
        import torch

        class aten_split_with_sizes(torch.nn.Module):
            def __init__(self):
                super(aten_split_with_sizes, self).__init__()                
                #self.sizes = 20

            def forward(self, x, y):
                return x.split([y.shape[0]], dim=0)

        ref_net = None

        return aten_split_with_sizes(), ref_net, ["aten::split_with_sizes", "prim::ListConstruct"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_split_with_sizes(self, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version, trace_model=True)

class TestSplitWithSizesCopy(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(20).astype(np.float32),np.random.randn(20).astype(np.float32))

    def create_model(self):
        import torch

        class aten_split_with_sizes_copy(torch.nn.Module):
            def __init__(self):
                super(aten_split_with_sizes_copy, self).__init__()                

            def forward(self, x, y):
                return torch.split_with_sizes_copy(x, [y.shape[0]], dim=0)

        ref_net = None

        return aten_split_with_sizes_copy(), ref_net, ["aten::split_with_sizes", "prim::ListConstruct"]

    @pytest.mark.precommit_fx_backend
    def test_split_with_sizes_copy(self, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version, trace_model=True)
