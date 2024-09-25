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


class TestMaxPool3D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, dilation, ceil_mode=True, dtype=torch.float32, return_indices=False):
        class aten_max_pool3d(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding, dilation, ceil_mode, return_indices) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.dtype = dtype
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices :
                    output, indices = torch.nn.functional.max_pool3d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation,self.ceil_mode, True)
                    return output, indices
                else :
                    output = torch.nn.functional.max_pool3d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, False)
                    return output, None
        ref_net = None

        if return_indices :
            return aten_max_pool3d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool3d_with_indices"
        return aten_max_pool3d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool3d"


    @pytest.mark.parametrize('input_shape', [[1, 3, 15, 15, 15], [3, 15, 15, 15]])
    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_max_pool3d(self, params, ceil_mode, dilation, return_indices, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, dilation=dilation, return_indices=return_indices),
                   ie_device, precision, ir_version, dynamic_shapes=True)
        
class TestMaxPool2D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, dilation, ceil_mode=True, dtype=torch.float32, return_indices=False):
        class aten_max_pool2d(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding, dilation, ceil_mode, return_indices) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.dtype = dtype
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices :
                    output, indices = torch.nn.functional.max_pool2d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation,self.ceil_mode, True)
                    return output, indices
                else :
                    output = torch.nn.functional.max_pool2d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, False)
                    return output, None
        ref_net = None

        if return_indices :
            return aten_max_pool2d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool2d_with_indices"
        return aten_max_pool2d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool2d"


    @pytest.mark.parametrize('input_shape', [[1, 3, 15, 15], [3, 15, 15]])
    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_max_pool2d(self, params, ceil_mode, dilation, return_indices, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, dilation=dilation, return_indices=return_indices),
                   ie_device, precision, ir_version, dynamic_shapes=True)
        
class TestMaxPool1D(PytorchLayerTest):
    
    def _prepare_input(self):
        return (self.input_tensor, )
    
    def create_model(self, kernel_size, stride, padding, dilation, ceil_mode=True, dtype=torch.float32, return_indices=False):
        class aten_max_pool1d(torch.nn.Module):
            def __init__(self, kernel_size, stride, padding, dilation, ceil_mode, return_indices) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.ceil_mode = ceil_mode
                self.dtype = dtype
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices :
                    output, indices = torch.nn.functional.max_pool1d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation,self.ceil_mode, True)
                    return output, indices
                else :
                    output = torch.nn.functional.max_pool1d(input_tensor, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, False)
                    return output, None
        ref_net = None

        if return_indices :
            return aten_max_pool1d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool1d_with_indices"
        return aten_max_pool1d(kernel_size, stride, padding, dilation, ceil_mode, return_indices), ref_net, "aten::max_pool1d"


    @pytest.mark.parametrize('input_shape', [[1, 3, 15], [3, 15]])
    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("dilation", [1, 2])
    @pytest.mark.parametrize("ceil_mode", [True, False])
    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_max_pool1d(self, params, ceil_mode, dilation, return_indices, ie_device, precision, ir_version, input_shape):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(**params, ceil_mode=ceil_mode, dilation=dilation, return_indices=return_indices),
                   ie_device, precision, ir_version, dynamic_shapes=True)
