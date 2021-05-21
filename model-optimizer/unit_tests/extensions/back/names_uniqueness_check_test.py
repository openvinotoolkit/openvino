# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.back.names_uniqueness_check import NamesUniquenessCheck
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

        ref_names = ['node_0', 'node_1', 'node']

        NamesUniquenessCheck().find_and_replace_pattern(graph)

        names = []
        for node in graph.get_op_nodes():
            names.append(node.name)

        self.assertEqual(names, ref_names)
