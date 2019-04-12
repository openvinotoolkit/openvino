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

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.layout import get_batch_dim, get_height_dim, get_width_dim, shape_for_layout
from mo.front.extractor import attr_getter
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class RegionYoloOp(Op):
    op = 'RegionYolo'

    def __init__(self, graph: Graph, attrs: Node):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': RegionYoloOp.regionyolo_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'coords',
            'classes',
            'num',
            'axis',
            'end_axis',
            'do_softmax',
            'anchors',
            'mask'
        ]

    def backend_attrs(self):
        return [
            'coords',
            'classes',
            'num',
            'axis',
            'end_axis',
            'do_softmax',
            ('anchors', lambda node: attr_getter(node, 'anchors')),
            ('mask', lambda node: attr_getter(node, 'mask'))
        ]

    @staticmethod
    def regionyolo_infer(node: Node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return
        axis = get_canonical_axis_index(input_shape, node.axis)
        end_axis = get_canonical_axis_index(input_shape, node.end_axis)
        node.axis = axis
        node.end_axis = end_axis
        if node.do_softmax:
            flat_dim = np.prod(input_shape[axis: end_axis + 1])
            node.out_node().shape = np.array([*input_shape[:axis], flat_dim, *input_shape[end_axis + 1:]])
        else:
            layout = node.graph.graph['layout']
            assert len(layout) == 4

            node.out_node().shape = shape_for_layout(layout,
                                                     batch=input_shape[get_batch_dim(layout, 4)],
                                                     features=(node.classes + node.coords + 1) * len(node.mask),
                                                     height=input_shape[get_height_dim(layout, 4)],
                                                     width=input_shape[get_width_dim(layout, 4)])
