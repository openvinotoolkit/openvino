# Copyright (C) 2018-2025 Intel Corporation
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
                                      stable=self.stable,
                                      dim=self.dim,
                                      descending=self.descending
                                      )[0]
                else:
                    return torch.sort(input_tensor,
                                      dim=self.dim,
                                      descending=self.descending
                                      )[0]

        ref_net = None
        return aten_sort(dim, descending, stable), ref_net, "aten::sort"

    @pytest.mark.parametrize("input_shape", [
        [16],
        [1, 4],
        [4, 4],
        [4, 4, 4],
        np.array([1, 2, 4, 6, 5, 8, 7]),
        np.array([6, 5, 4, 2, 3, 0, 1]),
        np.array([1, 1, 1, 2, 1, 3, 1, 4, 2, 5, 1, 2, 4, 4, 0]),
        np.array([[1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 2, 3], [1, 1, 1], [1, 2, 1]]),
        np.array([[9, 8, 8], [8, 7, 7], [7, 5, 6],
                  [8, 8, 9], [7, 7, 8], [6, 5, 7],
                  [8, 9, 8], [7, 8, 7], [5, 6, 7]]),
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]],
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]]]),
        np.array([[[3, 2, 2], [1, 2, 1], [3, 2, 2]],
                  [[1, 2, 1], [4, 3, 4], [3, 2, 2]],
                  [[3, 2, 2], [1, 2, 1], [7, 9, 9]]]),
        np.array([[[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]]])

    ])
    @pytest.mark.parametrize("descending", [
        True,
        False
    ])
    @pytest.mark.parametrize("stable", [
        False,
        None,
        True
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_sort(self, input_shape, descending, stable, ie_device, precision, ir_version):
        self.input_tensor = []
        if type(input_shape) is list:
            self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        else:
            self.input_tensor = input_shape
        dims = len(self.input_tensor.shape)
        for dim in range(-dims, dims):
            self._test(*self.create_model(dim, descending, stable),
                       ie_device, precision, ir_version)
