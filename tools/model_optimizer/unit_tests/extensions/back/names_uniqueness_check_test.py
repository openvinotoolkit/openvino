# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.names_uniqueness_check import NamesUniquenessCheck
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestNamesUniquenessCheck(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'cast': {'kind': 'op', 'op': 'Cast', 'name': 'node'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'cast'),
                ('cast', 'result')
            ]
        )

        NamesUniquenessCheck().find_and_replace_pattern(graph)
        names = [node.name for node in graph.get_op_nodes()]
        result_name = Node(graph, 'result').name

        self.assertTrue(len(set(names)) == 3)
        self.assertTrue(result_name == 'node')

    def test_2(self):
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node'},
                'cast': {'kind': 'op', 'op': 'Cast', 'name': 'node_0'},
                'result': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'cast'),
                ('cast', 'result')
            ]
        )

        NamesUniquenessCheck().find_and_replace_pattern(graph)
        names = [node.name for node in graph.get_op_nodes()]
        result_name = Node(graph, 'result').name

        self.assertTrue(len(set(names)) == 3)
        self.assertTrue(result_name == 'node')

    def test_3(self):
        graph = build_graph(
            nodes_attrs={
                'input': {'kind': 'op', 'op': 'Parameter', 'name': 'node_0'},
                'cast': {'kind': 'op', 'op': 'Cast', 'name': 'node_1'},
                'result_1': {'kind': 'op', 'op': 'Result', 'name': 'node'},
                'result_2': {'kind': 'op', 'op': 'Result', 'name': 'node'}
            },
            edges=[
                ('input', 'cast'),
                ('cast', 'result_1'),
                ('cast', 'result_2'),
            ]
        )
        NamesUniquenessCheck().find_and_replace_pattern(graph)
        names = [node.name for node in graph.get_op_nodes()]

        self.assertTrue('node' in names)
        self.assertTrue(len(set(names)) == 4)
