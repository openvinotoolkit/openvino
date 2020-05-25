"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.ATenToEmbeddingBag import AtenToEmbeddingBag
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'weights_inp': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'indices_inp': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'offsets_inp': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'aten': {'type': None, 'kind': 'op', 'op': 'ATen', 'mode': 0, 'operator': 'embedding_bag', 'name': 'my_aten'},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    # new EmbeddingBag layer
    'emb_bag': {'type': None, 'kind': 'op', 'op': 'ATenEmbeddingBag', 'mode': 0},
}


class AtenToEmbeddingBagTest(unittest.TestCase):
    def test(self):
        graph = build_graph(nodes_attributes,
                            [('weights_inp', 'aten', {'in': 0, 'out': 0}),
                             ('indices_inp', 'aten', {'in': 1, 'out': 0}),
                             ('offsets_inp', 'aten', {'in': 2, 'out': 0}),
                             ('aten', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('weights_inp', 'emb_bag', {'in': 0, 'out': 0}),
                                 ('indices_inp', 'emb_bag', {'in': 1, 'out': 0}),
                                 ('offsets_inp', 'emb_bag', {'in': 2, 'out': 0}),
                                 ('emb_bag', 'result', {'in': 0, 'out': 0}),
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = AtenToEmbeddingBag()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='ATenEmbeddingBag')[0]]['name'] == 'my_aten')
