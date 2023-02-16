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
    {'type': 'Gather'},
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
    {'type': 'Convolution'},
    {'type': 'MatMul'},
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

# TODO: Add attributes to GraphTransformer hw_config
TYPES_TO_QUANTIZABLE_PORTS = {'LSTMSequence': [0, 1, 4, 5], 'GRUSequence': [0, 1, 3, 4]}

ELTWISE_TYPES = ['Add', 'Multiply', 'Subtract', 'Divide', 'Less', 'LessEqual', 'Greater', 'GreaterEqual',
                 'Equal', 'NotEqual', 'FloorMod', 'LogicalOr', 'LogicalXor', 'LogicalAnd', 'Maximum', 'Minimum']


def is_eltwise(node):
    return node.type in ELTWISE_TYPES
