# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

def not_yet_supported(value):
    return pytest.param(
        value,
        marks = pytest.mark.xfail(
            reason="Failed due to aten::sargsort not yet supporting stable sorting. Ticket 105242"
        ),
    )

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

    @pytest.mark.parametrize("input_tensor", [
        np.random.rand(1, 4),
        np.random.rand(4, 4),
        np.random.rand(4, 4, 4),
        np.array([1, 2, 4, 6, 5, 8, 7]),
        np.array([6, 5, 4, 2, 3, 0, 1]),
        not_yet_supported(np.array([1, 1, 1, 2, 1, 3, 1, 4, 2, 5, 1, 2, 4, 4, 0])),
        not_yet_supported(np.array([[1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 1, 1], [1, 2, 1], [1, 2, 3],
                  [1, 2, 3], [1, 1, 1], [1, 2, 1]])),
        not_yet_supported(np.array([[9, 8, 8], [8, 7, 7], [7, 5, 6],
                  [8, 8, 9], [7, 7, 8], [6, 5, 7],
                  [8, 9, 8], [7, 8, 7], [5, 6, 7]])),
        not_yet_supported(np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]], 
                  [[5, 2, 4], [4, 9, 0], [7, 7, 9]]])),
        not_yet_supported(np.array([[[3, 2, 2], [1, 2, 1], [3, 2, 2]], 
                  [[1, 2, 1], [4, 3, 4], [3, 2, 2]], 
                  [[3, 2, 2], [1, 2, 1], [7, 9, 9]]])),
        not_yet_supported(np.array([[[2, 1, 3], [3, 2, 1], [1, 2, 3]], 
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]], 
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]], 
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]], 
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]],
                  [[2, 1, 3], [3, 2, 1], [1, 2, 3]], 
                  [[2, 0, 2], [1, 2, 1], [3, 2, 8]], 
                  [[3, 2, 2], [3, 2, 1], [1, 2, 3]]]))
    ])
    @pytest.mark.parametrize("descending", [
        True,
        False
    ])
    @pytest.mark.parametrize("stable", [
        False,
        None,
        not_yet_supported(True)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_argsort(self, input_tensor, descending, stable, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        dims = len(input_tensor.shape)
        for dim in range(-dims, dims):
            self._test(*self.create_model(dim, descending, stable), ie_device, precision, ir_version)
