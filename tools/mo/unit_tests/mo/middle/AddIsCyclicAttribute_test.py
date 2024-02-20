# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.AddIsCyclicAttribute import AddIsCyclicAttribute
from unit_tests.utils.graph import build_graph_with_attrs


class AddIsCyclicAttributeTest(unittest.TestCase):
    nodes = [('node_1', {}),
             ('node_2', {})]
    edges = [('node_1', 'node_2')]

    def test_1(self):
        """
        Acyclic case => graph.graph['is_cyclic'] should be False.
        """
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                       edges_with_attrs=self.edges)
        tested_pass = AddIsCyclicAttribute()
        tested_pass.find_and_replace_pattern(graph)

        assert graph.graph['is_cyclic'] is False

    def test_2(self):
        """
        Cyclic case => graph.graph['is_cyclic'] should be True.
        :return:
        """
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes,
                                       edges_with_attrs=self.edges,
                                       new_edges_with_attrs=[('node_2', 'node_1')])
        tested_pass = AddIsCyclicAttribute()
        tested_pass.find_and_replace_pattern(graph)

        assert graph.graph['is_cyclic'] is True
