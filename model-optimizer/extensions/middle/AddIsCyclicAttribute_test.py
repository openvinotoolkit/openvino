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

import unittest

from extensions.middle.AddIsCyclicAttribute import AddIsCyclicAttribute
from mo.utils.unittest.graph import build_graph_with_attrs


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
