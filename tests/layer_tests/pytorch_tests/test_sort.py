# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSortConstants(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, descending, stable):
        class aten_sort(torch.nn.Module):
            def __init__(self, dim, descending, stable) -> None:
                torch.nn.Module.__init__(self)
                self.stable = stable
                self.dim = dim
                self.descending = descending

            def forward(self, input_tensor):
                if self.stable is not None:
                    return torch.sort(input_tensor, 
                        stable = self.stable,
                        dim = self.dim, 
                        descending = self.descending
                    )[0]
                else:
                    return torch.sort(input_tensor, 
                        dim = self.dim, 
                        descending = self.descending
                    )[0]

        ref_net = None
        return aten_sort(dim, descending, stable), ref_net, "aten::sort"

    @pytest.mark.parametrize("input_tensor", [
        np.array([1, 2, 4, 6, 5]),
        np.array([0, 1] * 9),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]], [[5, 2, 4], [4, 9, 0], [7, 7, 9]]])
    ])
    @pytest.mark.parametrize("dim", [
        0,
        -1
    ])
    @pytest.mark.parametrize("descending", [
        True,
        False 
    ])
    @pytest.mark.parametrize("stable", [
        False,
        None
    ])
    def test_sort(self, input_tensor, dim, descending, stable, ie_device, precision, ir_version):
        self.input_tensor = input_tensor 
        self._test(*self.create_model(dim, descending, stable), ie_device, precision, ir_version)
