# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.shape import Shape


class InterpolateConcat(BackReplacementPattern):
    r"""
    Replaces hard-coded 1-port input of Interpolate with reshape-able sub-graph using the following Concat inputs

    BEFORE:
            input                   Const
    shape=[1, 3, 30, 40]      value=[60, 160]
            \                   /
           Interpolate(axes=(2, 3))     input_1
            shape=[1, 3, 60, 160]    shape=[1, 4, 60, 160]
                        \           /
                        Concat(axis=1)
                    shape=[1, 7, 60, 160]
    AFTER:
                input
            shape=[1, 3, 30, 40]           input_1
               |                     shape=[1, 4, 60, 160]
               |                      /        |
               |                  ShapeOf      |
               |                    |          |
               |               Gather          |
               |     indices=(2, 3); axis=0    |
               \                    |          |
                Interpolate(axes=(2, 3))      |
            shape=[1, 3, 60, 160]             |
                        \                   /
                           Concat(axis=1)
                        shape=[1, 7, 60, 160]
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].static_shape]
    force_shape_inference = True
    id = 'reshape_interpolate_through_concat'

    @staticmethod
    def make_interpolate_reshapeable(interpolate, concat):
        assert interpolate.soft_get('type') == 'Interpolate'
        assert concat.soft_get('type') == 'Concat'

        output_shape = interpolate.out_port(0).data.get_shape()

        interp_axes = [get_canonical_axis_index(output_shape, axis) for axis in Interpolate.get_axes(interpolate)]
        concat_axis = get_canonical_axis_index(output_shape, concat.axis)
        if concat_axis in interp_axes:
            return

        concat_srcs = [port.get_source() for port in concat.in_ports().values() if not port.disconnected()]
        non_interp_concat_srcs = [src for src in concat_srcs if src.node.soft_get('type') != 'Interpolate']
        if len(non_interp_concat_srcs) == 0:
            return

        graph = interpolate.graph
        src = non_interp_concat_srcs[0]

        shape = Shape(graph, {'name': src.node.soft_get('name', src.node.id) + '/Shape'}).create_node()
        shape.in_port(0).connect(src)
        gather = create_op_with_const_inputs(graph, Gather,
                                             {1: mo_array(interp_axes, dtype=np.int32), 2: int64_array(0)},
                                             {'name': shape.name + '/Gathered'}, shape)
        interpolate.in_port(1).get_connection().set_source(gather.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for interpolate in graph.get_op_nodes(type='Interpolate'):
            if interpolate.in_port(1).get_source().node.soft_get('type') != 'Const':
                continue
            dsts = interpolate.out_port(0).get_destinations()
            if len(dsts) == 1 and dsts[0].node.soft_get('type') == 'Concat':
                self.make_interpolate_reshapeable(interpolate, dsts[0].node)


class InterpolateReshapeWA(BackReplacementPattern):
    r"""
    Replaces hard-coded 1-port input of Interpolate with reshape-able sub-graph.
    WARNING: Could cause troubles if model has hard-coded Interpolate intentionally -- rare situation
    BEFORE:
        input                   Const
    shape=[1, 3, 30, 40]      value=[60, 160]
            \                   /
           Interpolate(axes=(2, 3))
            shape=[1, 3, 60, 160]
    AFTER:
            input
    shape=[1, 3, 30, 40]
        |                \
        |              ShapeOf
        |                |
        |              Gather                Const
        |        indices=(2, 3); axis=0    value=[2, 4]
        |                \                /
        |                    Multiply
        |                   /
    Interpolate(axes=(2, 3))
      shape=[1, 3, 60, 160]
    """
    enabled = False
    graph_condition = [lambda graph: not graph.graph['cmd_params'].static_shape]
    force_shape_inference = True
    id = 'reshape_interpolate_wa'

    def run_after(self):
        return [InterpolateConcat]

    @staticmethod
    def make_interpolate_reshapeable(interpolate):
        assert interpolate.soft_get('type') == 'Interpolate'
        axes = Interpolate.get_axes(interpolate)
        input_shape = interpolate.in_port(0).data.get_shape()
        output_shape = interpolate.out_port(0).data.get_shape()
        if not np.all(np.remainder(output_shape, input_shape) == 0) and \
                not np.all(np.remainder(input_shape, output_shape) == 0):
            return
        graph = interpolate.graph
        name = interpolate.soft_get('name', interpolate.id)
        shape = Shape(graph, {'name': name + '/ShapeOf'}).create_node()
        shape.in_port(0).connect(interpolate.in_port(0).get_source())
        gather = create_op_with_const_inputs(graph, Gather, {1: mo_array(axes, dtype=np.int32), 2: int64_array(0)},
                                             {'name': shape.name + '/Gathered'}, shape)
        multipliers = output_shape[axes] / input_shape[axes]
        mul = create_op_node_with_second_input(graph, Mul, multipliers, {'name': gather.name + '/Multiplied'}, gather)
        interpolate.in_port(1).get_connection().set_source(mul.out_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for interpolate in graph.get_op_nodes(type='Interpolate'):
            if interpolate.in_port(1).get_source().node.soft_get('type') == 'Const':
                self.make_interpolate_reshapeable(interpolate)
