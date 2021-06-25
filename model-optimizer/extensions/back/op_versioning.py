# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.utils.error import Error


class OpVersioning(BackReplacementPattern):
    enabled = False

    opset_1_types = set(map(lambda s: s.lower(), [
        "Abs",
        "Acos",
        "Add",
        "Asin",
        "Atan",
        "AvgPool",
        "BatchNormInference",
        "BinaryConvolution",
        "Broadcast",
        "CTCGreedyDecoder",
        "Ceiling",
        "Clamp",
        "Concat",
        "Const",  # leave this type not to change the whole IR reading infrastructure
        "Constant",
        "Convert",
        "ConvertLike",
        "Convolution",
        "ConvolutionBackpropData",
        "Cos",
        "Cosh",
        "DeformablePSROIPooling",
        "DepthToSpace",
        "DetectionOutput",
        "Divide",
        "Elu",
        "Equal",
        "Erf",
        "Exp",
        "FakeQuantize",
        "Floor",
        "FloorMod",
        "GRN",
        "Gather",
        "GatherTree",
        "Greater",
        "GreaterEqual",
        "GroupConvolution",
        "GroupConvolutionBackpropData",
        "HardSigmoid",
        "Interpolate",
        "LRN",
        "LSTMCell",
        "LSTMSequence",
        "Less",
        "LessEqual",
        "Log",
        "LogicalAnd",
        "LogicalNot",
        "LogicalOr",
        "LogicalXor",
        #"MVN",  # not really included into opset1
        "MatMul",
        "MaxPool",
        "Maximum",
        "Minimum",
        "Mod",
        "Multiply",
        "Negative",
        "NonMaxSuppression",
        "NormalizeL2",
        "NotEqual",
        "OneHot",
        "PReLU",
        "PSROIPooling",
        "Pad",
        "Parameter",
        "Power",
        "PriorBox",
        "PriorBoxClustered",
        "Proposal",
        #"ROIPooling",  # not really included into opset1
        "Range",
        "ReLU",
        "ReduceLogicalAnd",
        "ReduceLogicalOr",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "RegionYolo",
        #"ReorgYolo",  # not really included into opset1
        "Reshape",
        "Result",
        "ReverseSequence",
        "Select",
        "Selu",
        "ShapeOf",
        "Sigmoid",
        "Sign",
        "Sin",
        "Sinh",
        "Softmax",
        "SpaceToDepth",
        "Split",
        "Sqrt",
        "SquaredDifference",
        "Squeeze",
        "StridedSlice",
        "Subtract",
        "Tan",
        "Tanh",
        "TensorIterator",
        "Tile",
        "TopK",
        "Transpose",
        "Unsqueeze",
        "VariadicSplit",
    ]))

    opset_1_experimental_ops = set(map(lambda s: s.lower(), [
        "SimplerNMS",
        "SpatialTransformer",
        "ExperimentalDetectronGenerateProposalsSingleImage",
        "ExperimentalDetectronTopKROIs",
        "ExperimentalDetectronROIFeatureExtractor",
        "ExperimentalDetectronDetectionOutput",
        "ExperimentalDetectronPriorGridGenerator",
    ]))

    # Several ops were added to opset1 by mistake, now they are marked as belonging to opset2
    opset_2_legacy_ops = set(map(lambda s: s.lower(), [
        "MVN",
        "ReorgYolo",
        "ROIPooling",
    ]))

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            node_type = node.soft_get('type').lower()
            name = node.soft_get('name', node.id)

            if node.soft_get('version', None) == 'opset1' and node_type not in self.opset_1_types \
                                and node_type not in self.opset_2_legacy_ops:
                raise Error('Node {} has `version` attribute set to `{}`, but it is a reserved word, '
                            'please use another'.format(name, node.version))

            if not node.has_valid('version'):
                if node_type in self.opset_1_types:
                    node['version'] = 'opset1'
                elif node_type in self.opset_1_experimental_ops:
                    print("XXXXX")
                    print(node_type)
                    node['version'] = 'experimental'
                elif node_type in self.opset_2_legacy_ops:
                    node['version'] = 'opset2'
                else:
                    node['version'] = 'extension'
                    log.error('Please set `version` attribute for node {} with type={}'
                              ''.format(name, node.soft_get('type')), extra={'is_warning': True})
