# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.ForceStrictPrecision import ForceStrictPrecision
from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.strided_slice import StridedSlice


class CropToStridedSlice(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('crop', dict(op='Crop'))
            ],
            edges=[]
        )

    @staticmethod
    def mask_normalizer(shape_rank: int, axes: np.ndarray, values: np.ndarray):
        mask = np.zeros(shape_rank, dtype=np.int64)
        for i, axis in enumerate(axes):
            mask[axis] = values[i]
        return mask

    @staticmethod
    def list_to_ndarray(val):
        return mo_array(val) if mo_array(val).ndim != 0 else mo_array([val])

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        node = match['crop']
        assert node.has_valid('axis')
        node_axis = self.list_to_ndarray(node.axis)

        in_shape = node.in_port(0).data.get_shape()
        shape_rank = in_shape.size
        axis_mask = int64_array([1 if i in node_axis else 0 for i in range(shape_rank)])
        begin_mask = axis_mask.copy()
        end_mask = axis_mask.copy()

        ss = StridedSlice(graph, {'name': node.soft_get('name', node.id) + '/strided_slice', 'begin_mask': begin_mask,
                                  'end_mask': end_mask,
                                  'new_axis_mask': np.zeros(len(end_mask)),
                                  'shrink_axis_mask': np.zeros(len(end_mask)),
                                  'ellipsis_mask': np.zeros(len(end_mask))}).create_node()

        if len(node.in_nodes()) == 2 and node.has_valid('offset'):
            # Crop Type 1
            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node_axis, node.offset),
                                  'name': ss.name + '/begin'}).create_node()
            shape = Shape(graph, {'name': ss.name + '/shape_of_crop'}).create_node()
            end = Add(graph, {'name': ss.name + '/end'}).create_node()
            node.in_port(1).get_connection().get_source().connect(shape.in_port(0))
            node.in_port(1).disconnect()
            shape.out_port(0).connect(end.in_port(0))
            begin.out_port(0).connect(end.in_port(1))
        elif node.has_valid('dim') and node.has_valid('offset'):
            # Crop Type 2
            node_dim = self.list_to_ndarray(node.dim)
            node_offset = self.list_to_ndarray(node.offset)
            assert node_dim.size == node_offset.size == node_axis.size

            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node_axis, node_offset),
                                  'name': ss.name + '/begin'}).create_node()
            end_values = mo_array([node_offset[i] + node_dim[i] for i in range(len(node_dim))])
            end = Const(graph, {'value': self.mask_normalizer(shape_rank, node_axis, end_values),
                                'name': ss.name + '/end'}).create_node()
        elif node.has_valid('crop_begin') and node.has_valid('crop_end'):
            # Crop Type 3
            node_crop_begin = self.list_to_ndarray(node.crop_begin)
            node_crop_end = self.list_to_ndarray(node.crop_end)
            assert len(node_crop_begin) == len(node_crop_end) == len(node_axis)

            begin = Const(graph, {'value': self.mask_normalizer(shape_rank, node_axis, node_crop_begin),
                                  'name': ss.name + '/begin'}).create_node()
            shape = Shape(graph, {'name': ss.name + '/shape'}).create_node()

            end = Add(graph, {'name': ss.name + '/end'}).create_node()
            const = Const(graph, {'value': -1 * self.mask_normalizer(shape_rank, node_axis, node_crop_end),
                                  'name': ss.name + '/const'}).create_node()

            node.in_port(0).get_connection().get_source().connect(shape.in_port(0))
            shape.out_port(0).connect(end.in_port(0))
            const.out_port(0).connect(end.in_port(1))

        else:
            raise Exception("Unknown type of Crop")

        source = node.in_port(0).get_connection().get_source()

        stride = Const(graph, {'value': np.ones(shape_rank, dtype=np.int64),
                               'name': ss.name + '/stride'}).create_node()

        source.connect(ss.in_port(0))
        begin.out_port(0).connect(ss.in_port(1))
        end.out_port(0).connect(ss.in_port(2))
        stride.out_port(0).connect(ss.in_port(3))

        node.in_port(0).disconnect()
        node.out_port(0).get_connection().set_source(ss.out_port(0))

        ss['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}
