# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.reshape import Reshape


class DecomposeBias(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op=lambda op: op in ['Conv', 'ConvTranspose', 'Conv2D',
                                                            'Conv3D', 'Conv2DBackpropInput', 'MatMul',
                                                            'Conv3DBackpropInputV2', 'Convolution',
                                                            'Deconvolution', 'ConvND', 'Conv2D', 'Deconv2D']))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if node.has_port('in', 2) and not node.in_port(2).disconnected() and not node.has_and_set('shape_input'):
            bias_name = node.name
            new_node_name = node.name + '/WithoutBiases'
            add = Add(graph, dict(name=bias_name)).create_node()
            rename_nodes([(node, new_node_name), (add, bias_name)])
            node.out_port(0).get_connection().set_source(add.out_port(0))
            node.out_port(0).connect(add.in_port(0))
            node.in_port(2).get_connection().set_destination(add.in_port(1))

            bias = add.in_port(1).get_source().node
            if bias.has_valid("type") and bias.type == "Const":
                input_shape = add.in_port(0).data.get_shape()
                if len(input_shape) > 2:
                    dims_to_add = len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0
                    if dims_to_add > 0:
                        reshape = create_op_node_with_second_input(
                            graph, Reshape, int64_array([input_shape[1]] + [1] * dims_to_add),
                            {'name': node.id + '/Dims'})
                        add.in_port(1).get_connection().set_destination(reshape.in_port(0))
                        reshape.out_port(0).connect(add.in_port(1))
