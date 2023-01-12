# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


@pytest.mark.parametrize('input_data', [(np.random.randn(2, 3, 2), np.array(2), np.array(6)),
                                        (np.random.randn(4), np.array(2), np.array(2))])
class TestViewListConstruct(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_data

    def create_model(self):
        class aten_view_list_construct(torch.nn.Module):

            def forward(self, input_tensor, dim1: int, dim2: int):
                return input_tensor.view(dim1, dim2)

        ref_net = None

        return aten_view_list_construct(), ref_net, "aten::view"

    @pytest.mark.nightly
    def test_view_list_construct(self, ie_device, precision, ir_version, input_data):
        self.input_data = input_data
        self._test(*self.create_model(), ie_device, precision, ir_version)


@pytest.mark.parametrize('input_data', [(np.random.randn(2, 3, 2), 2, 6),
                                        (np.random.randn(4), 2, 2)])
class TestView(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_data[0],)

    def create_model(self):

        class aten_view(torch.nn.Module):

            def __init__(self, input_data) -> None:
                super().__init__()
                self.dim1 = input_data[1]
                self.dim2 = input_data[2]

            def forward(self, input_tensor):
                return input_tensor.view(self.dim1, self.dim2)

        ref_net = None

        return aten_view(self.input_data), ref_net, "aten::view"

    @pytest.mark.nightly
    def test_view(self, ie_device, precision, ir_version, input_data):
        self.input_data = input_data
        self._test(*self.create_model(), ie_device, precision, ir_version)
