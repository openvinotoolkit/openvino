# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestBincount(PytorchLayerTest):

    def _prepare_input(self, input_data_info, with_weights, minlength):
        input_data, i_dtype, w_dtype = input_data_info
        inputs = (np.array(input_data, dtype=i_dtype),)
        if with_weights:
            weights = np.ones(len(input_data), dtype=w_dtype)
            for i in range(len(input_data)):
                weights[i] = (i % 5) * 1.5
            inputs += (weights,)
        return inputs

    def create_model(self, with_weights, minlength):
        class BincountNoWeights(torch.nn.Module):
            def __init__(self, minlength):
                super().__init__()
                self.minlength = minlength

            def forward(self, x):
                return torch.bincount(x, minlength=self.minlength)

        class BincountWithWeights(torch.nn.Module):
            def __init__(self, minlength):
                super().__init__()
                self.minlength = minlength

            def forward(self, x, w):
                return torch.bincount(x, weights=w, minlength=self.minlength)

        if with_weights:
            return BincountWithWeights(minlength), "aten::bincount"
        return BincountNoWeights(minlength), "aten::bincount"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("with_weights", [True, False])
    @pytest.mark.parametrize("minlength", [0, 5, 10])
    @pytest.mark.parametrize("input_data_info", [
        ([0, 1, 1, 2, 2, 2, 3], np.int32, np.float32),
        ([0, 0, 0, 0], np.int64, np.float64),
        ([0, 1, 2, 3, 4], np.int32, np.int64),
        ([3, 3, 3, 3, 3], np.int64, np.int32),
    ])
    def test_bincount(self, with_weights, minlength, input_data_info, ie_device, precision, ir_version):
        self._test(
            *self.create_model(with_weights, minlength),
            ie_device,
            precision,
            ir_version,
            dynamic_shapes=False,
            kwargs_to_prepare_input={
                "input_data_info": input_data_info,
                "with_weights": with_weights,
                "minlength": minlength,
            },
        )
