"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
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
        "DeformableConvolution",
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
        "MVN",
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
        "ROIPooling",
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
        "ReorgYolo",
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
        "RNNCell",
        "GRUCell",
        "SimplerNMS",
        "SpatialTransformer",
        "ExperimentalDetectronGenerateProposalsSingleImage",
        "ExperimentalDetectronTopKROIs",
        "ExperimentalDetectronROIFeatureExtractor",
        "ExperimentalDetectronDetectionOutput",
        "ExperimentalDetectronPriorGridGenerator",
    ]))

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            node_type = node.soft_get('type').lower()
            name = node.soft_get('name', node.id)

            if node.soft_get('version', None) == 'opset1' and node_type not in self.opset_1_types:
                raise Error('Node {} has `version` attribute set to `opset1`, but it is a reserved word, '
                            'please use another'.format(name))

            if not node.has_valid('version'):
                if node_type in self.opset_1_types:
                    node['version'] = 'opset1'
                elif node_type in self.opset_1_experimental_ops:
                    node['version'] = 'experimental'
                else:
                    node['version'] = 'extension'
                    log.error('Please set `version` attribute for node {} with type={}'
                              ''.format(name, node.soft_get('type')), extra={'is_warning': True})
