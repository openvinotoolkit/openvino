# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_shapes',
[
    [
        [2, 3, 2], np.array(2), np.array(6)
    ],
    [
        [4], np.array(2), np.array(2)
    ]
])
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
    @pytest.mark.precommit
    def test_view_list_construct(self, ie_device, precision, ir_version, input_shapes):
        self.input_data = []
        for input_shape in input_shapes:
            if type(input_shape) is list:
                self.input_data.append(np.random.randn(*input_shape).astype(np.float32))
            else:
                self.input_data.append(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version)

@pytest.mark.parametrize('input_shapes',
[
    [
        [4], np.array(2)
    ]
])
class TestViewDtype(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_data

    def create_model(self):
        class aten_view_dtype(torch.nn.Module):

            def forward(self, input_tensor, dtype):
                return input_tensor.view(torch.int64)

        ref_net = None

        return aten_view_dtype(), ref_net, "aten::view"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_view_dtype(self, ie_device, precision, ir_version, input_shapes):
        self.input_data = []
        for input_shape in input_shapes:
            if type(input_shape) is list:
                self.input_data.append(np.random.randn(*input_shape).astype(np.float32))
            else:
                self.input_data.append(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version)


@pytest.mark.parametrize('input_shapes',
[
    [
        [4], [2, 2]
    ]
])
class TestViewSize(PytorchLayerTest):

    def _prepare_input(self):
        return self.input_data

    def create_model(self):
        class aten_view_size(torch.nn.Module):

            def forward(self, input_tensor, input_size):
                return input_tensor.view(input_size.size()[:])

        ref_net = None

        return aten_view_size(), ref_net, "aten::view"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_view_size(self, ie_device, precision, ir_version, input_shapes):
        self.input_data = []
        for input_shape in input_shapes:
            if type(input_shape) is list:
                self.input_data.append(np.random.randn(*input_shape).astype(np.float32))
            else:
                self.input_data.append(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version)

@pytest.mark.parametrize('input_shapes',
[
    [
        [2, 3, 2], 2, 6
    ],
    [
        [4], 2, 2
    ],
    [
        [4], 2, 2.1
    ]
])
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
                return input_tensor.view(self.dim1, int(self.dim2))

        ref_net = None

        return aten_view(self.input_data), ref_net, "aten::view"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_view(self, ie_device, precision, ir_version, input_shapes):
        self.input_data = []
        for input_shape in input_shapes:
            if type(input_shape) is list:
                self.input_data.append(np.random.randn(*input_shape).astype(np.float32))
            else:
                self.input_data.append(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version)

@pytest.mark.parametrize('input_shapes',
[
    [
        [2, 3, 2], 2, 6
    ],
    [
        [4], 2, 2
    ],
    [
        [4], 2, 2.1
    ]
])
class TestViewCopy(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_data[0],)

    def create_model(self):
        class aten_view_copy(torch.nn.Module):

            def __init__(self, input_data) -> None:
                super().__init__()
                self.dim1 = input_data[1]
                self.dim2 = input_data[2]

            def forward(self, input_tensor):
                return torch.view_copy(input_tensor, [self.dim1, int(self.dim2)])

        ref_net = None

        return aten_view_copy(self.input_data), ref_net, "aten::view_copy"

    @pytest.mark.precommit_fx_backend
    def test_view_copy(self, ie_device, precision, ir_version, input_shapes):
        self.input_data = []
        for input_shape in input_shapes:
            if type(input_shape) is list:
                self.input_data.append(np.random.randn(*input_shape).astype(np.float32))
            else:
                self.input_data.append(input_shape)
        self._test(*self.create_model(), ie_device, precision, ir_version)
