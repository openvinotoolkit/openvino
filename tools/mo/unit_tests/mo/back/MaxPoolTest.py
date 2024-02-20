# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.MaxPool import MaxPool
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class TestMaxPool(unittest.TestCase):

    def test_no_out_normalization(self):
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'input_data': {'kind': 'data'},
                'pool': {'kind': 'op', 'name': 'node', 'type': 'Pooling', 'pool_method': 'max'},
                'pool_data': {'kind': 'data'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'input_data'),
                ('input_data', 'pool'),
                ('pool', 'pool_data'),
                ('pool_data', 'result')
            ]
        )

        graph_ref = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'input_data': {'kind': 'data'},
                'pool': {'kind': 'op', 'name': 'node', 'type': 'MaxPool'},
                'pool_data': {'kind': 'data'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'},
            },
            edges=[
                ('input', 'input_data'),
                ('input_data', 'pool'),
                ('pool', 'pool_data'),
                ('pool_data', 'result'),
            ]
        )

        MaxPool().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_out_normalization(self):
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'input_data': {'kind': 'data'},
                'pool': {'kind': 'op', 'name': 'node', 'type': 'Pooling', 'pool_method': 'max'},
                'pool_data': {'kind': 'data'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'input_data'),
                ('input_data', 'pool'),
                ('pool', 'pool_data'),
                ('pool_data', 'result')
            ]
        )

        graph_ref = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'input_data': {'kind': 'data'},
                'pool': {'kind': 'op', 'name': 'node', 'type': 'MaxPool'},
                'pool_data': {'kind': 'data'},
                'pool_data_added': {'kind': 'data'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'},
                'result_added': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'input_data'),
                ('input_data', 'pool'),
                ('pool', 'pool_data'),
                ('pool_data', 'result'),
                ('pool', 'pool_data_added'),
                ('pool_data_added', 'result_added')
            ]
        )

        pool_op = Node(graph, 'pool')
        pool_op.add_output_port(1)      # add disconnected output port to check normalization

        MaxPool().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
