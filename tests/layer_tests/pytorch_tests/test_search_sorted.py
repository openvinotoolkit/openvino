# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest
import numpy as np


class TestSearchSorted(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.sorted).astype(self.sorted_type),np.array(self.values).astype(self.values_type))

    def create_model(self, right_mode):
        import torch

        class aten_searchsorted(torch.nn.Module):
            def __init__(self, right_mode):
                super(aten_searchsorted, self).__init__()
                self.right_mode = right_mode

            def forward(self, sorted, values):
                return torch.searchsorted(sorted, values, right=self.right_mode)

        ref_net = None

        return aten_searchsorted(right_mode), ref_net, "aten::searchsorted"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("sorted", "values"), [
            ([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], [[3, 6, 9], [3, 6, 9]]),
            ([1, 3, 5, 7, 9], [[3, 6, 9],[0, 5, 20]]),
            ([4091, 4092], [[4091, 4092]]), # fp16 cannot exactly represent 4091 number 
            ([1.23, 2.99], [[1.355, 2.9991]])     
        ])
    @pytest.mark.parametrize("right_mode", [False, True])
    @pytest.mark.parametrize("sorted_type", [np.float32, np.float16, np.int8])
    @pytest.mark.parametrize("values_type", [np.float16, np.int32, np.int64])
    def test_searchsorted(self, sorted, values, right_mode, sorted_type, values_type, ie_device, precision, ir_version):
        self.sorted = sorted
        self.values = values
        self.sorted_type = sorted_type
        self.values_type = values_type
        if ie_device == "CPU" and sorted_type == np.float16 and sorted == [4091, 4092]:
            pytest.skip(reason="CPU plugin on defult converts fp16 to fp32, if that happens the test will fail for those malicious values")
        self._test(*self.create_model(right_mode), ie_device, precision, ir_version)
