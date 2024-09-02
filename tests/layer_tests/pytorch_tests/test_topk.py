# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTopK(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, k, dim, largest, sort):
        import torch

        class aten_topk(torch.nn.Module):
            def __init__(self, k, dim, largest, sort):
                super(aten_topk, self).__init__()
                self.k = k
                self.dim = dim
                self.largest = largest
                self.sort = sort

            def forward(self, input_tensor):
                if self.dim is None:
                    return torch.topk(input_tensor, k=self.k, largest=self.largest, sorted=self.sort)
                else:
                    return torch.topk(input_tensor, k=self.k, dim=self.dim, largest=self.largest, sorted=self.sort)
        ref_net = None

        return aten_topk(k, dim, largest, sort), ref_net, "aten::topk"

    @pytest.mark.parametrize(("input_shape"), [
        [7, 5, 5, 4],
        [5, 6, 6, 7, 8]
    ])

    @pytest.mark.parametrize(("k"), [
        3,
        1,
        2,
    ])

    @pytest.mark.parametrize(("dim"), [
        0,
        2,
        -1,
        None,
    ])

    @pytest.mark.parametrize(("largest"), [
        True,
        False,
    ])
    # For False it is hard to test because in Pytorch implementation
    # there is not promise on the order of output values
    @pytest.mark.parametrize(("sort"), [
        True,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_topK(self, input_shape, k, dim, largest, sort, ie_device, precision, ir_version):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(k, dim, largest, sort), ie_device, precision, ir_version)
