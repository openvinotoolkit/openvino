"""
 Copyright (c) 2018-2020 Intel Corporation

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

import unittest

import numpy as np
from generator import generator, generate

from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.ir_reader.layer_to_class import groupconv_to_conv
from mo.utils.unittest.graph import build_graph


@generator
class TestFunction(unittest.TestCase):

    @generate(*[([1, 32, 112, 112], [32, 1, 1, 3], [32, 1, 1, 1, 3], 32),
                ([1, 32, 112, 112], [32, 1, 1, 1, 3], None, 32),
                ])
    def test_groupconv_to_conv(self, shape, weights_shape, reshape_shape, group):

        weights_const = np.random.randn(*weights_shape).astype(np.float32)

        nodes_attributes = {
            'input': {'kind': 'op', 'type': 'Parameter'},
            'input_data': {'shape': shape, 'kind': 'data'},

            'group_conv': {'kind': 'op', 'type': 'GroupConvolution'},
            'group_conv_data': {'shape': shape, 'kind': 'data'},

            'conv': {'kind': 'op', 'type': 'Convolution', 'group': group},
            'conv_data': {'shape': shape, 'kind': 'data'},

            'weights': {'kind': 'op', 'type': 'Const', 'value': weights_const},
            'weights_data': {'shape': weights_shape, 'kind': 'data'},

            'reshape': {'kind': 'op', 'type': 'Reshape'},
            'reshape_data': {'shape': reshape_shape, 'kind': 'data'},
            'reshape_const': {'kind': 'op', 'type': 'Const'},
            'reshape_const_data': {'shape': len(reshape_shape) if reshape_shape is not None else None, 'kind': 'data'},

            'add': {'kind': 'op', 'type': 'Add'},
            'add_data': {'shape': shape, 'kind': 'data'},
            'add_const': {'kind': 'op', 'type': 'Const'},
            'add_const_data': {'shape': [1, 32, 1, 1], 'kind': 'data'},
            'result': {'kind': 'op', 'type': 'Result'}
        }

        edges = [('input', 'input_data'),
                 ('input_data', 'group_conv'),
                 ('weights', 'weights_data'),
                 ('group_conv', 'group_conv_data'),
                 ('group_conv_data', 'add'),
                 ('add_const', 'add_const_data'),
                 ('add_const_data', 'add'),
                 ('add', 'add_data'),
                 ('add_data', 'result'),
                 ]

        if reshape_shape is not None:

            edges += [('weights_data', 'reshape'),
                      ('reshape_const', 'reshape_const_data'),
                      ('reshape_const_data', 'reshape'),
                      ('reshape', 'reshape_data'),
                      ('reshape_data', 'group_conv')]
        else:
            edges.append(('weights_data', 'group_conv'))

        graph = build_graph(nodes_attributes, edges, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'input_data'),
                                 ('input_data', 'conv'),
                                 ('weights', 'weights_data'),
                                 ('weights_data', 'conv'),
                                 ('conv', 'conv_data'),
                                 ('conv_data', 'add'),
                                 ('add_const', 'add_const_data'),
                                 ('add_const_data', 'add'),
                                 ('add', 'add_data'),
                                 ('add_data', 'result'),
                                 ], nodes_with_edges_only=True)

        for op in graph.get_op_nodes(type='GroupConvolution'):
            groupconv_to_conv(op)

        if reshape_shape is None:
            new_shape = [weights_shape[1] * group, *weights_shape[2:]]
            weights_const = np.reshape(weights_const, new_shape)
            node = Node(graph_ref, 'weights')
            node.value = weights_const

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
