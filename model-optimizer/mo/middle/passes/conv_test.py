"""
 Copyright (c) 2018 Intel Corporation

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

from mo.middle.passes.conv import convert_muladd_to_scaleshift_or_power
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ScaleShift layer
    'scaleshift_1': {'type': 'ScaleShift', 'value': None, 'kind': 'op', 'op': 'ScaleShift'},
    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'value': None, 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1': {'value': None, 'kind': 'op', 'op': 'Add'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Power layer
    'power_1': {'type': 'Power', 'kind': 'op', 'op': 'Power', 'scale': None, 'shift': None, 'power': None},
    'power_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class MulAddToScaleShiftOrPower(unittest.TestCase):
    def _create_graph_with_mul_add(self, mul_w, add_w):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_data', 'add_1'),
                             ('add_1_w', 'add_1'),
                             ('add_1', 'add_1_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mul_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'add_1_data': {'shape': np.array([1, 227, 227, 3]), 'is_output': True},
                             'mul_1_w': {'shape': np.array(mul_w.shape) if mul_w is not None else None,
                                         'value': np.array(mul_w) if mul_w is not None else None},
                             'add_1_w': {'shape': np.array(add_w.shape) if add_w is not None else None,
                                         'value': np.array(add_w) if add_w is not None else None},
                             })
        del graph['mul_1']['mul_1_data'][0]['in']
        del graph['add_1']['add_1_data'][0]['in']
        return graph

    def test_mul_add_to_scaleshift_1(self):
        graph = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array([1, 2, 3]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'scaleshift_1_data'),
                                 ],
                                {'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'scaleshift_1_data': {'is_output': True}
                                 })

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    def test_mul_add_to_power_1(self):
        graph = self._create_graph_with_mul_add(np.array([3]), np.array([2]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'power_1'),
                                 ('power_1', 'power_1_data'),
                                 ],
                                {'power_1': {'scale': 3, 'shift': 2, 'power': 1},
                                 'power_1_data': {'is_output': True}
                                 })

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'power_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_1(self):
        graph = self._create_graph_with_mul_add(None, np.array([2]))
        graph_ref = self._create_graph_with_mul_add(None, np.array([2]))

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_2(self):
        graph = self._create_graph_with_mul_add(np.array([2]), None)
        graph_ref = self._create_graph_with_mul_add(np.array([2]), None)

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_3(self):
        graph = self._create_graph_with_mul_add(None, None)
        graph_ref = self._create_graph_with_mul_add(None, None)

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_4(self):
        graph = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array([3]))
        graph_ref = self._create_graph_with_mul_add(np.array([1, 2, 3]), np.array(3))

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mul_add_neg_5(self):
        graph = self._create_graph_with_mul_add(np.array([3]), np.array([3, 2, 1]))
        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'scaleshift_1'),
                                 ('scaleshift_1_w', 'scaleshift_1'),
                                 ('scaleshift_1_b', 'scaleshift_1'),
                                 ('scaleshift_1', 'add_1_data'),
                                 ],
                                {'scaleshift_1_w': {'shape': np.array([3]), 'value': np.array([3, 3, 3])},
                                 'scaleshift_1_b': {'shape': np.array([3]), 'value': np.array([3, 2, 1])},
                                 'add_1_data': {'is_output': True}
                                 })

        convert_muladd_to_scaleshift_or_power(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'add_1_data', 'add_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


if __name__ == '__main__':
    unittest.main()
