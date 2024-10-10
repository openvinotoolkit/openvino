# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestSearchSorted(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.array(self.sorted).astype(np.float32),np.array(self.values).astype(np.float32))

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
    @pytest.mark.parametrize(("sorted", "values"), [([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], [[3, 6, 9], [3, 6, 9]]),([1, 3, 5, 7, 9], [[3, 6, 9],[0, 5, 20]])])
    @pytest.mark.parametrize("right_mode", [False, True])
    def test_searchsorted(self, sorted, values, right_mode, ie_device, precision, ir_version):
        self.sorted = sorted
        self.values = values
        self._test(*self.create_model(right_mode), ie_device, precision, ir_version)
