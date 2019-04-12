"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.op import Op


class Flatten(Op):
    op = 'Flatten'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis', 'end_axis']

    @staticmethod
    def infer(node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            log.debug('The input shape for the layer "{}" is not defined'.format(node.soft_get('name')))
            return

        axis = get_canonical_axis_index(input_shape, node.axis)
        end_axis = node.end_axis if node.has('end_axis') else -1
        end_axis = get_canonical_axis_index(input_shape, end_axis)
        prod_axes = np.prod(input_shape[axis: end_axis + 1])
        node.out_node(0).shape = int64_array([*input_shape[0: axis], prod_axes, *input_shape[end_axis + 1:]])

        if node.in_node().has_valid('value'):
            node.out_node().value = node.in_node().value.copy().reshape(node.out_node(0).shape)
