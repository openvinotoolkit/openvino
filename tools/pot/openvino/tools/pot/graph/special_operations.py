# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

QUANTIZE_AGNOSTIC_OPERATIONS = [
    {'type': 'MaxPool'},
    {'type': 'ReduceMax'},
    {'type': 'Reshape'},
    {'type': 'Concat'},
    {'type': 'Flatten'},
    {'type': 'Squeeze'},
    {'type': 'Unsqueeze'},
    {'type': 'Split'},
    {'type': 'VariadicSplit'},
    {'type': 'Crop'},
    {'type': 'Transpose'},
    {'type': 'Tile'},
    {'type': 'StridedSlice'},
    {'type': 'ShuffleChannels'},
    {'type': 'Broadcast'},
    {'type': 'Pad'},
    {'type': 'Minimum'},
    {'type': 'Maximum'},
    {'type': 'ConvertLike'},
    {'type': 'DepthToSpace'}
]

OPERATIONS_WITH_BIAS = [
    {'type': 'Convolution'},
    {'type': 'MatMul'}
]

OPERATIONS_CHANNEL_AXIS = {'Convolution': 1, 'MatMul': -1}

OPERATIONS_WITH_WEIGHTS = [
    {'type': 'Convolution'},
    {'type': 'ConvolutionBackpropData'},
    {'type': 'GroupConvolution'},
    {'type': 'GroupConvolutionBackpropData'},
    {'type': 'MatMul'},
]


CONCAT_UNIFY_OUTPUTS = [
    {'type': 'ConvolutionBackpropData'},
    {'type': 'Convolution'}
]

CONCAT_UNIFY_INPUTS = [
    {'type': 'FakeQuantize'},
    {'type': 'Concat'}
]

TRANSPOSED_OPERATIONS = [
    {'type': 'ConvolutionBackpropData'}
]

SPLIT_OPERATIONS = [
    {'type': 'VariadicSplit'},
    {'type': 'Split'}
]

DETECTION_OUTPUT_FINAL_TYPES = [
    {'type': 'NonMaxSuppression'},
    {'type': 'TopK'}
]

ELTWISE_TYPES = ['Add', 'Multiply', 'Subtract', 'Divide', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
                 'Equal', 'NotEqual', 'FloorMod', 'LogicalOr', 'LogicalXor', 'LogicalAnd', 'Maximum', 'Minimum']

ELTWISE_ADD_SUB = [
    {'type': 'Add'},
    {'type': 'Subtract'}
]


def is_eltwise(node):
    return node.type in ELTWISE_TYPES
