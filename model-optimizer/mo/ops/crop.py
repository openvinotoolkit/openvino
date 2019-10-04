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
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Crop(Op):
    op = 'Crop'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('axis', lambda node: None if not node.has_valid('axis') else ','.join(map(str, node.axis))),
            ('offset', lambda node: None if not node.has_valid('offset') else ','.join(map(str, node.offset))),

            ('dim', lambda node: None if not node.has_valid('dim') else ','.join(map(str, node.dim))),

            ('crop_begin', lambda node: None if not node.has_valid('crop_begin') else ','.join(map(str, node.crop_begin))),
            ('crop_end', lambda node: None if not node.has_valid('crop_end') else ','.join(map(str, node.crop_end))),
        ]

    @staticmethod
    def infer(node: Node):
        """
        Crops the shape of the output blob according to input ones be specified params.
        Detailed Crop description can be found in IR Catalog specification.
        In short: crop layer can be represented in three ways:
            1. Two inputs, where the shape of the second input is crop dim (axis and offset attrs)
            2. One input and dim, axis and offset attributes.
            3. Ont input and axis, crop_begin and crop_end attributes
        """

        input_count = len(node.in_nodes())

        if input_count == 2:
            Crop._two_inputs_infer(node)
        elif input_count == 1:
            Crop._one_input_infer(node)
        else:
            log.error('Wrong number of input tensors ({}) in {}'.format(input_count, node.name))
            return

    @staticmethod
    def _one_input_infer(node: Node):
        input_shape = np.array(node.in_node().shape)

        if input_shape is None:
            log.error('input_shape is none for {} node'.format(node.name))
            return

        if not node.has_valid('axis'):
            log.error('axis attribute is missing for {} node. should be set in crop extractor'.format(node.name))
            return

        output_shape = input_shape
        if node.has_valid('dim'):
            if len(node.dim) != len(node.axis):
                log.error('number of axis should match number of dim')
                return
            output_shape[node.axis] = node.dim
        elif node.has_valid('crop_begin') and node.has_valid('crop_end'):
            if len(node.crop_begin) != len(node.axis) or len(node.crop_end) != len(node.axis):
                log.error('number of crop_begin/crop_end should match number of axis')
                return
            if type(node.axis) in [list, tuple]:
                for i in range(len(node.axis)):
                    output_shape[node.axis[i]] = output_shape[node.axis[i]] - node.crop_begin[i] - node.crop_end[i]
            else:
                output_shape[node.axis] = output_shape[node.axis] - node.crop_begin - node.crop_end
        else:
            log.error('Crop node {} should have either dim or crop_begin and crop_end attributes'.format(node.name))
            return

        node.out_node().shape = np.array(output_shape)
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

    @staticmethod
    def _two_inputs_infer(node: Node):
        N = len(node.in_nodes())

        shapes = [node.in_node(i).shape for i in range(N)]
        if any(s is None for s in shapes):
            log.error('Not all input shapes were defined for {} node'.format(node.name))
            return

        if not node.has_valid('axis'):
            log.error('axis attribute is missing for {} node. should be set in crop extractor'.format(node.name))
            return

        if not node.has_valid('offset'):
            log.error('offset attribute is missing for {} node. should be set in crop extractor'.format(node.name))
            return

        input_shape = np.array(shapes[0])
        start_axis = get_canonical_axis_index(input_shape, node.axis)
        node.axis = start_axis

        reference_shape = np.array(shapes[1])
        if node.has_valid('axes'):
            '''
            The axes parameter  contain shape indexes for second input and 
            show which shape indexes we need to use for dim attribute.
            '''
            input_dim = node.axes
            node.in_port(1).disconnect()
        else:
            input_dim = list(range(0, input_shape.size))

        # set new shape to current shape
        new_shape = input_shape.copy()
        ir_axis = []
        ir_offset = []
        dim = []

        for i in input_dim:
            if i < start_axis:
                new_shape[i] = input_shape[i]
                continue

            crop_offset = 0
            if len(node.offset) == 1:
                crop_offset = node.offset[0]
            elif len(node.offset) > 1:
                crop_offset = node.offset[i - start_axis]

            if input_shape[i] - crop_offset < reference_shape[i]:
                log.error('The crop for dimension is out of bounds in ' + node.node)
                return

            dim.append(reference_shape[i])
            ir_axis.append(i)
            ir_offset.append(crop_offset)
            new_shape[i] = reference_shape[i]

        node.axis = ir_axis
        node.offset = ir_offset
        node['dim'] = dim
        node.out_node().shape = new_shape
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
