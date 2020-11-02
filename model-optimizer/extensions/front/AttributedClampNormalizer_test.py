"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.front.AttributedClampNormalizer import AttributedClampNormalizer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'attr_clamp': {'type': 'Clamp', 'kind': 'op', 'op': 'AttributedClamp', 'name': 'attr_clamp',
                   'min': np.array(-3.5, dtype=np.float32), 'max': np.array(3.5, dtype=np.float32)},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new Clamp layer and inputs
    'clamp': {'type': None, 'kind': 'op', 'op': 'Clamp'},
    **const('min', np.array(-3.5, dtype=np.float32)),
    **const('max', np.array(3.5, dtype=np.float32)),
}


class AttributedClampNormalizerTest(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'attr_clamp', {'in': 0, 'out': 0}),
                             ('attr_clamp', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'clamp', {'in': 0, 'out': 0}),
                                 ('min', 'clamp', {'in': 1, 'out': 0}),
                                 ('max', 'clamp', {'in': 2, 'out': 0}),
                                 ('clamp', 'result')
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AttributedClampNormalizer()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Clamp')[0]]['name'] == 'attr_clamp')
