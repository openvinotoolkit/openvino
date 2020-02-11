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

from mo.graph.graph import Node
from mo.middle.passes.conv import convert_muladd_to_scaleshift, convert_add_or_mul_to_scaleshift
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'value': None, 'kind': 'op', 'op': 'ScaleShift'},
    'const_scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'op'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'const_scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'op'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'value': None, 'kind': 'op', 'op': 'Mul'},
    'const_mul_1_w': {'value': None, 'shape': None, 'kind': 'op'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1': {'value': None, 'kind': 'op', 'op': 'Add'},
    'const_add_1_w': {'value': None, 'shape': None, 'kind': 'op'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'op_output': {'kind': 'op', 'op': 'Result'},
}


class MulAddToScaleShift(unittest.TestCase):
    def _create_graph_with_mul_add(self, mul_w, add_w):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array(mul_w.shape) if mul_w is not None else None,
                                               'value': np.array(mul_w) if mul_w is not None else None},
                             'mul_1_w': {'shape': np.array(mul_w.shape) if mul_w is not None else None,
                                         'value': np.array(mul_w) if mul_w is not None else None},
                             'const_add_1_w': {'shape': np.array(add_w.shape) if add_w is not None else None,
                                               'value': np.array(add_w) if add_w is not None else None},
                             'add_1_w': {'shape': np.array(add_w.shape) if add_w is not None else None,
                                         'value': np.array(add_w) if add_w is not None else None},
                             })
        del graph['mul_1']['mul_1_data'][0]['in']
        del graph['add_1']['add_1_data'][0]['in']
        return graph

    @unittest.skip("ScaleShift is not supported")
    def test_mul_add_to_scaleshift_1(self):
        graph = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array([1, 2, 3]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('const_scaleshift_1_w', 'scaleshift_1_w'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('const_scaleshift_1_b', 'scaleshift_1_b'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'scaleshift_1_data'),
                                 ('scaleshift_1_data', 'op_output'),
                                 ],
                                {'const_scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'const_scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'scaleshift_1_data': {}
                                 })

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    @unittest.skip("Power is not supported")
    def test_mul_add_neg_1(self):
        graph = self._create_graph_with_mul_add(None, np.array([2]))
        graph_ref = self._create_graph_with_mul_add(None, np.array([2]))

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_2(self):
        graph = self._create_graph_with_mul_add(np.array([2]), None)
        graph_ref = self._create_graph_with_mul_add(np.array([2]), None)

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_3(self):
        graph = self._create_graph_with_mul_add(None, None)
        graph_ref = self._create_graph_with_mul_add(None, None)

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    @unittest.skip("TODO investigate why this test is not passing")
    def test_mul_add_neg_4(self):
        graph = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array([3]))
        graph_ref = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array([3]))

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    @unittest.skip("ScaleShift is not supported")
    def test_mul_add_neg_5(self):
        graph = self._create_graph_with_mul_add(np.array([3]), np.array([3, 2, 1]))
        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('const_scaleshift_1_w', 'scaleshift_1_w'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('const_scaleshift_1_b', 'scaleshift_1_b'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'add_1_data'),
                                 ('add_1_data', 'op_output'),
                                 ],
                                {'const_scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([3, 3, 3])},
                                 'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([3, 3, 3])},
                                 'const_scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 })

        convert_muladd_to_scaleshift(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


class AddToScaleShift(unittest.TestCase):
    @staticmethod
    def _create_graph_with_add(add_w: np.ndarray):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'add_1'),
                             ('const_add_1_w', 'add_1_w'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ('add_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_add_1_w': {'shape': np.array(add_w.shape) if add_w is not None else None,
                                               'value': np.array(add_w) if add_w is not None else None},
                             'add_1_w': {'shape': np.array(add_w.shape) if add_w is not None else None,
                                         'value': np.array(add_w) if add_w is not None else None},
                             }, nodes_with_edges_only=True)
        del graph['add_1']['add_1_data'][0]['in']
        return graph

    @staticmethod
    def _create_graph_with_mul(mul_w: np.ndarray):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('const_mul_1_w', 'mul_1_w'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_1_w': {'shape': np.array(mul_w.shape) if mul_w is not None else None,
                                               'value': np.array(mul_w) if mul_w is not None else None},
                             'mul_1_w': {'shape': np.array(mul_w.shape) if mul_w is not None else None,
                                         'value': np.array(mul_w) if mul_w is not None else None},
                             }, nodes_with_edges_only=True)
        del graph['mul_1']['mul_1_data'][0]['in']
        return graph

    @unittest.skip("ScaleShift is not supported")
    def test_add_to_scaleshift_1(self):
        graph = AddToScaleShift._create_graph_with_add(np.array([1, 2, 3], dtype=np.float32))
        graph.stage = 'middle'

        graph_ref = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3])},

                             'const_scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 1, 1])},

                             'const_scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             }, nodes_with_edges_only=True)

        convert_add_or_mul_to_scaleshift(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)

        scsh_node = Node(graph, 'op_output').in_port(0).get_source().node

        self.assertTrue(graph.get_edge_data(scsh_node.in_node(1).id, scsh_node.id)[0]['bin'] == 'weights')
        self.assertTrue(graph.get_edge_data(scsh_node.in_node(2).id, scsh_node.id)[0]['bin'] == 'biases')

    @unittest.skip("ScaleShift is not supported")
    def test_mul_to_scaleshift_1(self):
        graph = AddToScaleShift._create_graph_with_mul(np.array([1, 2, 3], dtype=np.float32))
        graph.stage = 'middle'

        graph_ref = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('const_scaleshift_1_w', 'scaleshift_1_w'),
                             ('const_scaleshift_1_b', 'scaleshift_1_b'),
                             ('scaleshift_1_w', 'scaleshift_1'),
                             ('scaleshift_1_b', 'scaleshift_1'),
                             ('scaleshift_1', 'scaleshift_1_data'),
                             ('scaleshift_1_data', 'op_output')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'scaleshift_1_data': {'shape': np.array([1, 227, 227, 3])},

                             'const_scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                             'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},

                             'const_scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([0, 0, 0])},
                             }, nodes_with_edges_only=True)

        convert_add_or_mul_to_scaleshift(graph)
        graph.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)

        scsh_node = Node(graph, 'op_output').in_port(0).get_source().node

        self.assertTrue(graph.get_edge_data(scsh_node.in_node(1).id, scsh_node.id)[0]['bin'] == 'weights')
        self.assertTrue(graph.get_edge_data(scsh_node.in_node(2).id, scsh_node.id)[0]['bin'] == 'biases')



if __name__ == '__main__':
    unittest.main()
