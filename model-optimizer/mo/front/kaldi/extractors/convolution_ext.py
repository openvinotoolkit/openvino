"""
 Copyright (c) 2018 Intel Corporation

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

import copy

import numpy as np

from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.convolution import Convolution
from mo.ops.op import Op


class Convolution1DFrontExtractor(FrontExtractorOp):
    op = 'convolution'
    enabled = True

    @staticmethod
    def extract(node: Node) -> bool:
        params = node.pb
        mapping_rule = {
            'output': params.output,
            'patch_stride': params.patch_stride,
            'bias_term': None,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
            'dilation': int64_array([1, 1, 1, 1]),
            'kernel': int64_array([1, 1, 1, params.kernel]),
            'stride': int64_array([1, 1, 1, params.stride]),
            'kernel_spatial': int64_array([1, params.kernel]),
            'input_feature_channel': 1,
            'output_feature_channel': 0,
            'kernel_spatial_idx': [2,3],
            'group': 1,
            'reshape_kernel': True,
        }
        mapping_rule.update(layout_attrs())
        mapping_rule.update(weights_biases(params.bias_term, params))
        if len(params.blobs) > 1 and len(params.blobs[1]) > 0:
            mapping_rule['bias_addable'] = True
        else:
            mapping_rule['bias_addable'] = False

        Op.get_op_class_by_name('Convolution').update_node_stat(node, mapping_rule)
        return __class__.enabled
