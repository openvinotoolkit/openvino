# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_tensor',
[
    [2, 1, 3], [3, 7], [1, 1, 4, 4]
])
class TestLen(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = self.input_tensor * 10
        return (input_tensor.astype(np.int64),)

    def create_model(self):
        class aten_len(torch.nn.Module):

            def forward(self, input_tensor):
                return torch.tensor(len(input_tensor))

        ref_net = None

        return aten_len(), ref_net, "aten::len"

    def create_model_int_list(self):
        class aten_len(torch.nn.Module):

            def forward(self, input_tensor):
                int_list = input_tensor.size()
                return torch.tensor(len(int_list))

        ref_net = None

        return aten_len(), ref_net, "aten::len"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_len(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = np.random.randn(*input_tensor).astype(np.float32)
        self._test(*self.create_model(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_len_int_list(self, ie_device, precision, ir_version, input_tensor):
        self.input_tensor = np.random.randn(*input_tensor).astype(np.float32)
        self._test(*self.create_model_int_list(),
                   ie_device, precision, ir_version, use_convert_model=True)


class TestLenEmpty(PytorchLayerTest):

    def _prepare_input(self):
        input_tensor = np.random.randn(1, 2, 3) * 10
        return (input_tensor.astype(np.int64),)

    def create_model_empty(self):
        class aten_len(torch.nn.Module):

            def forward(self, input_tensor):
                # len of empty slice
                return torch.tensor(len(input_tensor[0:0]))

        ref_net = None

        return aten_len(), ref_net, "aten::len"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_len_empty(self, ie_device, precision, ir_version):
        self._test(*self.create_model_empty(),
                   ie_device, precision, ir_version)
