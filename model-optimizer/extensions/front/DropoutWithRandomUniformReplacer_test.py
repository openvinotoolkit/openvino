# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import unittest

from extensions.front.DropoutWithRandomUniformReplacer import DropoutWithRandomUniformReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op


class DropoutWithRandomUniformReplacerTest(unittest.TestCase):
    def test(self):
        nodes = {
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('shape', {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'}),
            **regular_op('random_uniform', {'type': 'RandomUniform', 'kind': 'op', 'op': 'RandomUniform',
                                            'name': 'dropout/RU'}),
            **regular_op('mul', {'type': 'Mul', 'kind': 'op', 'op': 'Mul'}),
            **regular_op('add', {'type': 'Add', 'kind': 'op', 'op': 'Add'}),
            **regular_op('add2', {'type': 'Add', 'kind': 'op', 'op': 'Add'}),
            **regular_op('floor', {'type': 'Floor', 'kind': 'op', 'op': 'Floor'}),
            'add_const': {'kind': 'op', 'op': 'Const', 'value': np.array(0.0), 'data_type': np.float32},
            **result('result'),

            # new nodes to be added
            'broadcast_const': {'kind': 'op', 'op': 'Const', 'value': np.array(0.5), 'data_type': np.float32},
            **regular_op('broadcast', {'type': 'Broadcast', 'kind': 'op', 'op': 'Broadcast'}),
        }
        edges = [('input', 'shape'),
                 ('shape', 'random_uniform'),
                 ('random_uniform', 'mul'),
                 ('mul', 'add'),
                 ('add_const', 'add'),
                 ('add', 'add2'),
                 ('add2', 'floor'),
                 ('floor', 'result')]
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        DropoutWithRandomUniformReplacer().find_and_replace_pattern(graph)

        edges_ref = [('input', 'shape'),
                     ('broadcast_const', 'broadcast'),
                     ('shape', 'broadcast'),
                     ('broadcast', 'mul'),
                     ('mul', 'add'),
                     ('add_const', 'add'),
                     ('add', 'add2'),
                     ('add2', 'floor'),
                     ('floor', 'result')]
        graph_ref = build_graph(nodes, edges_ref, nodes_with_edges_only=True)

        # check graph structure after the transformation and output name
        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Broadcast')[0]]['name'] == 'dropout/RU')
