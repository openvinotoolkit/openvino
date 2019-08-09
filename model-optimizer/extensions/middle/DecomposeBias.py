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

from extensions.ops.elementwise import Add
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class DecomposeBias(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op=lambda op: op in ['Conv', 'ConvTranspose', 'Conv2D',
                                                            'Conv3D', 'Conv2DBackpropInput', 'MatMul', 'FullyConnected',
                                                            'Conv3DBackpropInputV2', 'Convolution',
                                                            'Deconvolution', 'ConvND', 'Conv2D', 'Deconv2D']))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if node.has_port('in', 2) and not node.in_port(2).disconnected():
            add = Add(graph, dict(name=node.name + "/Bias")).create_node()
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
                            graph, Reshape, np.array([input_shape[1]] + [1] * dims_to_add, dtype=np.int64),
                            {'name': node.id + '/Dims'})
                        add.in_port(1).get_connection().set_destination(reshape.in_port(0))
                        reshape.out_port(0).connect(add.in_port(1))
