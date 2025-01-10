# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestArgSort(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, dim, descending, stable):
        class aten_argsort(torch.nn.Module):
            def __init__(self, dim, descending, stable) -> None:
                torch.nn.Module.__init__(self)
                self.dim = dim
                self.descending = descending
                self.stable = stable

            def forward(self, input_tensor):
                if self.stable is not None:
                    return torch.argsort(input_tensor,
                        dim = self.dim,
                        descending = self.descending,
                        stable = self.stable
                    )
                else:
                    return torch.argsort(input_tensor,
                        dim = self.dim,
                        descending = self.descending
                    )
        ref_net = None

        return aten_argsort(dim, descending, stable), ref_net, "aten::argsort"

    @pytest.mark.parametrize("tensor_stable_pair", [
        ([1, 4], False),
        ([4, 4], False),
        ([4, 4, 4], False),
        (np.array([1, 2, 4, 6, 5, 8, 7]), False),
        (np.array([6, 5, 4, 2, 3, 0, 1]), False),
        (np.array([1, 1, 1, 2, 1, 3, 1, 4, 2, 5, 1, 2, 4, 4, 0]), True),
        (np.array([[1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 2, 3], [1, 1, 1], [1, 2, 1]]), True),
        (np.array([[9, 8, 8], [8, 7, 7], [7, 5, 6],
                  [8, 8, 9], [7, 7, 8], [6, 5, 7],
                  [8, 9, 8], [7, 8, 7], [5, 6, 7]]), True),
        (np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]],
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]]]), True),
        (np.array([[[3, 2, 2], [1, 2, 1], [3, 2, 2]],
                  [[1, 2, 1], [4, 3, 4], [3, 2, 2]],
                  [[3, 2, 2], [1, 2, 1], [7, 9, 9]]]), True),
        (np.array([[[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]],
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]],
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]]]), True)
    ])
    @pytest.mark.parametrize("descending", [
        True,
        False
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_argsort(self, tensor_stable_pair, descending, ie_device, precision, ir_version):
        input_shape, stable = tensor_stable_pair
        if type(input_shape) is list:
            self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        else:
            self.input_tensor = input_shape
        dims = len(self.input_tensor.shape)
        for dim in range(-dims, dims):
            stable_values = [True] if stable else [True, False, None]
            for stable_value in stable_values:
                self._test(*self.create_model(dim, descending, stable_value), ie_device, precision, ir_version)
