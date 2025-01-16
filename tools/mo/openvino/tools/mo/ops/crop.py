# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op, PermuteAttrs
from openvino.tools.mo.utils.error import Error


class Crop(Op):
    op = 'Crop'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('axis', lambda node: None if not node.has_valid('axis') else ','.join(map(str, node.axis))),
            ('offset', lambda node: None if not node.has_valid('offset') else ','.join(map(str, node.offset))),

            ('dim', lambda node: None if not node.has_valid('dim') else ','.join(map(str, node.dim))),

            ('crop_begin', lambda node: None if not node.has_valid('crop_begin') else ','.join(map(str,
                                                                                                   node.crop_begin))),
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
        input_shape = node.in_port(0).data.get_shape()
        node_name = node.soft_get('name', node.id)
        if input_shape is None:
            raise Error('input_shape is none for {} node'.format(node_name))

        if not node.has_valid('axis'):
            raise Error('axis attribute is missing for {} node. should be set in crop extractor'.format(node_name))

        output_shape = input_shape.copy()
        if node.has_valid('dim'):
            if len(node.dim) != len(node.axis):
                raise Error('Number of axis "{}" should match number of dim "{}" for node "{}"'
                            ''.format(node.axis, node.dim, node_name))
            output_shape[node.axis] = node.dim
        elif node.has_valid('crop_begin') and node.has_valid('crop_end'):
            if len(node.crop_begin) != len(node.axis) or len(node.crop_end) != len(node.axis):
                raise Error('number of crop_begin({})/crop_end({}) should match number of axis "{}" for node "{}"'
                            ''.format(node.crop_begin, node.crop_end, node.axis, node_name))
            if type(node.axis) in [list, tuple]:
                for i in range(len(node.axis)):
                    output_shape[node.axis[i]] = output_shape[node.axis[i]] - node.crop_begin[i] - node.crop_end[i]
            else:
                output_shape[node.axis] = output_shape[node.axis] - node.crop_begin - node.crop_end
        else:
            raise Error('Crop node {} should have either dim or crop_begin and crop_end attributes'.format(node_name))

        node.out_port(0).data.set_shape(output_shape)
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

    @staticmethod
    def _two_inputs_infer(node: Node):
        N = len(node.in_nodes())
        node_name = node.soft_get('name', node.id)

        shapes = [node.in_port(i).data.get_shape() for i in range(N)]
        if any(s is None for s in shapes):
            raise Error('Not all input shapes were defined for {} node'.format(node_name))

        if not node.has_valid('axis'):
            raise Error('axis attribute is missing for {} node. should be set in crop extractor'.format(node_name))

        if not node.has_valid('offset'):
            raise Error('offset attribute is missing for {} node. should be set in crop extractor'.format(node_name))

        input_shape = shapes[0].copy()
        start_axis = get_canonical_axis_index(input_shape, node.axis)
        node.axis = start_axis

        reference_shape = shapes[1].copy()
        if node.has_valid('axes'):
            # The axes parameter  contain shape indexes for second input and show which shape indexes we need to use for
            # dim attribute.
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
                raise Error('The crop for dimension is out of bounds in node {}'.format(node_name))

            dim.append(reference_shape[i])
            ir_axis.append(i)
            ir_offset.append(crop_offset)
            new_shape[i] = reference_shape[i]

        node.axis = ir_axis
        node.offset = ir_offset
        node['dim'] = dim
        node.out_port(0).data.set_shape(new_shape)

        if node.in_node(0).has_valid('value') and \
                not getattr(node.graph.graph['cmd_params'], 'enable_ssd_gluoncv', False):
            out_value = np.copy(node.in_node(0).value)

            slice_indexes = []
            for s in out_value.shape:
                slice_indexes.append(slice(0, s))

            for axis in input_dim:
                slice_indexes[axis] = slice(0, new_shape[axis])
                out_value = out_value[tuple(slice_indexes)]
            node.out_port(0).data.set_value(out_value)

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
