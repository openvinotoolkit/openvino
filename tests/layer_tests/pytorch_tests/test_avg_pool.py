# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest
import numpy as np


d2_params = [{'kernel_size': [3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': 1},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [0, 1]},
             {'kernel_size': [3, 3], 'stride': [1, 1], 'padding': [1, 0]},
             {'kernel_size': [3, 3], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [2, 1], 'padding': 0},
             {'kernel_size': [2, 1], 'stride': None, 'padding': 0},
             {'kernel_size': [2, 1], 'stride': [], 'padding': 0},
             ]

d1_params = [{'kernel_size': 3, 'stride': 1, 'padding': 0},
             {'kernel_size': (4,), 'stride': 1, 'padding': 1},
             {'kernel_size': 4, 'stride': (5,), 'padding': 2},
             {'kernel_size': 4, 'stride': None, 'padding': 0},
             ]

d3_params = [{'kernel_size': [3, 3, 3], 'stride': 1, 'padding': 0},
             {'kernel_size': [3, 3, 3], 'stride': [1, 1, 1], 'padding': 1},
             {'kernel_size': [3, 3, 3], 'stride': [3, 3, 3], 'padding': [0, 0, 0]},
             {'kernel_size': [3, 2, 1], 'stride': [3, 1, 1], 'padding': [0, 0, 0]},
             ]


class TestAvgPool3D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, ceil_mode=True, count_include_pad=True):
        class aten_avg_pool3d(torch.nn.Module):

            def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                print(kernel_size)
                self.stride = stride
                print(stride)
                self.padding = padding
                self.ceil_mode = ceil_mode
                self.count_include_pad = count_include_pad

            def forward(self, input_tensor):
                return torch.nn.functional.avg_pool3d(input_tensor, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)
        ref_net = None

        return aten_avg_pool3d(kernel_size, stride, padding, ceil_mode=True, count_include_pad=True), ref_net, "aten::avg_pool3d"


    @pytest.mark.parametrize('input_shape', [[1, 3, 15, 15, 15], [3, 15, 15, 15]])
    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_avg_pool3d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, dynamic_shapes=True)
    
        
        



class TestAvgPool2D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, ceil_mode=True, count_include_pad=True):
        class aten_avg_pool2d(torch.nn.Module):

            def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                print(kernel_size)
                self.stride = stride
                print(stride)
                self.padding = padding
                self.ceil_mode = ceil_mode
                self.count_include_pad = count_include_pad

            def forward(self, input_tensor):
                return torch.nn.functional.avg_pool2d(input_tensor, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)
        ref_net = None

        return aten_avg_pool2d(kernel_size, stride, padding, ceil_mode=True, count_include_pad=True), ref_net, "aten::avg_pool2d"

    @pytest.mark.parametrize('input_shape', [[1, 3, 15, 15], [3, 15, 15]])    
    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_avg_pool2d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True)



class TestAvgPool1D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, ceil_mode=True, count_include_pad=True):
        class aten_avg_pool1d(torch.nn.Module):

            def __init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.ceil_mode = ceil_mode
                self.count_include_pad = count_include_pad

            def forward(self, input_tensor):
                return torch.nn.functional.avg_pool1d(input_tensor, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                                                      self.count_include_pad)
        ref_net = None

        return aten_avg_pool1d(kernel_size, stride, padding, ceil_mode=True, count_include_pad=True), ref_net, "aten::avg_pool1d"
    @pytest.mark.parametrize('input_shape', [[1, 3, 15], [3, 15]])
    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize("count_include_pad", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_avg_pool1d(self, params, ceil_mode, count_include_pad, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, count_include_pad=count_include_pad),
                   ie_device, precision, ir_version, trace_model=True, dynamic_shapes=False)
        



