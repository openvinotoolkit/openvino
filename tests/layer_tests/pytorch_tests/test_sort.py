# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSortConstants(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, descending):
        import torch

        class aten_sort(torch.nn.Module):
            def __init__(self, dim, descending):
                super(aten_sort, self).__init__()
                self.dim = dim
                self.descending = descending 

            def forward(self, x):
                return torch.sort(x, dim=self.dim, descending=self.descending)[0]

        ref_net = None

        return aten_sort(dim, descending), ref_net, "aten::sort"

    @pytest.mark.parametrize("input_tensor", [
        np.array([1, 2, 4, 6, 5]),
        np.array([0, 1] * 9),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]]])
    ])
    @pytest.mark.parametrize("dim", [
        # 0, 1, -1,
        0, -1
    ])
    @pytest.mark.parametrize("descending", [
        True,
        False 
    ])
    # @pytest.mark.parametrize("stable", [
    #     # For False there is no guarantee for the order
    #     # so there is no good way of testing that

    # ])

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sort(self, input_tensor, dim, descending, ie_device, precision, ir_version):
        self.input_tensor = input_tensor 
        if ie_device == "CPU":
            self._test(*self.create_model(dim, descending), ie_device, precision, ir_version)
