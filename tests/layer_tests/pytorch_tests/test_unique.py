# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestUnique2(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, sorted, return_inverse, return_counts):
        import torch

        class aten_unique2(torch.nn.Module):
            def __init__(self, sorted, return_inverse, return_counts):
                super(aten_unique2, self).__init__()
                self.op = torch._unique2
                self.sorted = sorted
                self.return_inverse = return_inverse
                self.return_counts = return_counts

            def forward(self, x):
                result, inverse, counts = self.op(x, self.sorted, self.return_inverse, self.return_counts)
                results = (result, )
                if self.return_inverse:
                    results += (inverse, )
                if self.return_counts:
                    results += (counts, )
                return results

        ref_net = None
        model_class, op = (aten_unique2, "aten::_unique2")

        return model_class(sorted, return_inverse, return_counts), ref_net, op

    @pytest.mark.parametrize("input_shape", [
        [4], [2, 3], [5, 4, 6], [3, 7, 1, 4], [16, 3, 32, 32]
    ])
    @pytest.mark.parametrize("sorted", [False, True])
    @pytest.mark.parametrize("return_inverse", [False, True])
    @pytest.mark.parametrize("return_counts", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_unique2(self, input_shape, sorted, return_inverse, return_counts, ie_device, precision, ir_version):
        self.input_tensor = np.random.randint(0, 10, size=input_shape).astype(np.int32)
        self._test(*self.create_model(sorted, return_inverse, return_counts), ie_device, precision, ir_version)
