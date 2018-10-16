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
from mo.front.common.partial_infer.convolution import calc_convolution_caffe
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
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
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'dilation': int64_array([1, 1, 1, 1]),
            'kernel': int64_array([1, 1, params.kernel, 1]),
            'stride': int64_array([1, 1, params.stride, 1]),
            'infer': Convolution1DFrontExtractor.infer
        }
        mapping_rule.update(layout_attrs())
        mapping_rule.update(weights_biases(params.bias_term, params))
        if len(params.blobs) > 1 and len(params.blobs[1]) > 0:
            mapping_rule['bias_addable'] = True
        else:
            mapping_rule['bias_addable'] = False

        Op.get_op_class_by_name('Convolution').update_node_stat(node, mapping_rule)
        return __class__.enabled

    @staticmethod
    def infer(node: Node) -> None:
        input_shape = node.in_node().shape
        output_shape = copy.copy(input_shape)
        output_shape[node.batch_dims] = input_shape[node.batch_dims]
        output_shape[node.channel_dims] = node.output
        output_shape[node.spatial_dims] = calc_convolution_caffe(input_shape[node.spatial_dims],
                                                                 np.flip(node.stride[node.spatial_dims], axis=0),
                                                                 np.add.reduce(node.pad, axis=0),
                                                                 np.flip(node.kernel[node.spatial_dims], axis=0))
        node.out_node().shape = output_shape
