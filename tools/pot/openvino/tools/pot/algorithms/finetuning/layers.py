# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from openvino.tools.pot.graph import node_utils as nu
from openvino.tools.pot.utils.logger import get_logger
from .utils import get_weight_node

logger = get_logger(__name__)


# pylint: disable=W0221
class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data, val_min, val_max):
        ctx.save_for_backward(input_data)
        ctx.val_min = val_min
        ctx.val_max = val_max
        return input_data.round()

    @staticmethod
    def backward(ctx, grad_output):
        (input_data,) = ctx.saved_tensors
        alpha = 0.01
        mask = (input_data <= ctx.val_max) & (input_data >= ctx.val_min)
        mask = mask.type(input_data.dtype)
        grad_input = grad_output * (mask * (1 - alpha) + alpha)
        return grad_input, None, None


# pylint: disable=E1102,W0223
class FakeQuantize(torch.nn.Module):
    """
    A pytorch wrapper for a single FakeQuantize node.
    """

    @staticmethod
    def is_able_to_wrap(node):
        if node.type != 'FakeQuantize':
            return False
        is_const = [
            node.in_port(i).get_source().node.type == 'Const' for i in range(1, 5)
        ]
        if not all(is_const):
            return False
        data = [node.in_port(i).data.get_value() for i in range(1, 5)]
        diff = [np.max(np.abs(data[i] - data[i + 2])) for i in [0, 1]]
        diff = max(diff)
        if diff > 10 ** -8:
            logger.info('FakeQuantize {} has different input and output scales'.format(node.name))
            return False

        return True

    def __init__(self, node, device='cpu', asymmetric=False):
        super(FakeQuantize, self).__init__()
        self.node = node
        self.device = device
        input_0 = nu.get_node_input(self.node, 0)
        self.is_weight_fq = input_0.type == 'Const'
        self.asymmetric = asymmetric

        min_val = nu.get_node_value(nu.get_node_input(self.node, 1))
        max_val = nu.get_node_value(nu.get_node_input(self.node, 2))
        min_val = np.array(min_val, dtype=np.float32)
        self.min = torch.tensor(min_val).to(self.device)
        self.min = torch.nn.Parameter(self.min) if self.asymmetric else self.min

        ranges = np.array(max_val - min_val, dtype=np.float32)
        self.scale = torch.tensor(ranges).log()
        self.scale = self.scale.to(self.device)
        self.scale = torch.nn.Parameter(self.scale)

        self.val_h = int(self.node.levels - 1)
        self.val_l = 0

    def update_node_params(self):
        scale = self.scale.exp()
        max_level = scale.detach().cpu().numpy()
        max_level = np.reshape(max_level, nu.get_input_shape(self.node, 2))
        min_level = self.min.detach().cpu().numpy()
        min_level = np.reshape(min_level, nu.get_input_shape(self.node, 1))
        max_level = min_level + max_level

        self.node.in_port(1).data.set_value(min_level)
        self.node.in_port(2).data.set_value(max_level)
        self.node.in_port(3).data.set_value(min_level)
        self.node.in_port(4).data.set_value(max_level)

    def forward(self, x):
        scale = self.scale.exp()
        s = self.val_h * scale.reciprocal()
        x = x - self.min
        x = x * s
        x = x.clamp(max=self.val_h, min=self.val_l)
        x = STERound.apply(x, self.val_l, self.val_h)
        x = x * s.reciprocal() + self.min
        return x


# pylint: disable=E1102,W0223
class LinearModule(torch.nn.Module):
    """
    A pytorch wrapper for a single Conv2d/Linear node.
    """

    @staticmethod
    def is_able_to_wrap(node):
        if node.type not in ['Convolution', 'MatMul', 'GroupConvolution']:
            return False

        node_weight = nu.get_node_input(node, 1)
        if node_weight.type == 'FakeQuantize':
            node_weight = nu.get_node_input(node_weight, 0)
        if node_weight.type != 'Const':
            return False

        if node.type != 'MatMul':

            weights = nu.get_node_value(node_weight)
            if len(weights.shape) != 4:
                return False

            s = node.stride
            stride_check = (s[2] == s[3])

            d = node.dilation
            dilation_check = (d[2] == d[3])

            if not dilation_check or not stride_check:
                return False

        bias_node = nu.get_bias_for_node(node)
        if bias_node is not None:
            bias_value = nu.get_node_value(bias_node)
            if bias_value.shape[0] != 1:
                return False
        return True

    def __init__(self,
                 node,
                 input_fq=None,
                 wrap_weight_fq=False,
                 device='cpu',
                 set_quantized_values_to_weight_parameter=False,
                 asymmetric=False):
        super().__init__()

        self.node = node
        self.device = device

        self.set_quantized_values_to_weight_parameter = set_quantized_values_to_weight_parameter
        self.weight_fq, self.input_fq = None, input_fq

        if wrap_weight_fq:
            weight_fq = nu.get_node_input(self.node, 1)
            weight_fq_wrapper = FakeQuantize
            if not weight_fq_wrapper.is_able_to_wrap(weight_fq):
                logger.warning('Was not able to wrap layer %s with pytorch', weight_fq.name)
                self.weight_fq = None
            else:
                self.weight_fq = weight_fq_wrapper(weight_fq, device=device,
                                                   asymmetric=asymmetric)

        node_weight = get_weight_node(node)
        weights = nu.get_node_value(node_weight)
        self.weights_dtype = weights.dtype
        weights = torch.from_numpy(weights).to(torch.float32)
        weights = weights.to(device)
        self.weights = torch.nn.Parameter(weights)

        self.bias = None
        bias_node = nu.get_bias_for_node(self.node)
        if bias_node is not None:
            bias = nu.get_node_value(bias_node)
            self.bias_dtype = bias.dtype
            bias = torch.from_numpy(bias).to(torch.float32).squeeze()
            bias = bias if bias.shape else bias.reshape(1)
            bias = bias.to(device)
            self.bias = torch.nn.Parameter(bias)

        if self.node.type != 'MatMul':
            self.stride = (int(node.stride[2]), int(node.stride[3]))
            self.pads_begin, self.pads_end = node.pad[2], node.pad[3]
            self.dilation = (int(node.dilation[2]), int(node.dilation[3]))
            self.group = 1 if 'group' not in node else int(node.group)

    def update_node_params(self):
        weights = self.weights.detach()
        weights = weights.cpu() if self.device != 'cpu' else weights
        weights = weights.numpy().astype(self.weights_dtype)
        weight_node = get_weight_node(self.node)
        nu.set_node_value(weight_node, weights)

        if self.weight_fq is not None:
            self.weight_fq.update_node_params()
        if self.input_fq is not None:
            self.input_fq.update_node_params()

        if self.bias is not None:
            bias_node = nu.get_bias_for_node(self.node)
            bias_shape = nu.get_node_value(bias_node).shape
            bias = self.bias.data.reshape(bias_shape)

            bias = bias.detach()
            bias = bias.cpu() if self.device != 'cpu' else bias
            bias = bias.numpy().astype(self.bias_dtype)
            nu.set_node_value(bias_node, bias)

    def forward(self, x):
        w = self.weight_fq(self.weights) if self.weight_fq is not None else self.weights
        x = self.input_fq(x) if self.input_fq is not None else x
        if self.set_quantized_values_to_weight_parameter and self.weight_fq is not None:
            self.weights.data = w

        if self.node.type == 'MatMul':
            x = torch.nn.functional.linear(x,
                                           self.weights,
                                           bias=self.bias)
        else:
            pad_top, pad_bottom = int(self.pads_begin[0]), int(self.pads_begin[1])
            pad_left, pad_right = int(self.pads_end[0]), int(self.pads_end[1])
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            x = torch.nn.functional.conv2d(
                x,
                self.weights,
                bias=self.bias,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.group
            )

        return x
