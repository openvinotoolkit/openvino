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

from mo.middle.passes.eliminate import graph_clean_up
from mo.middle.passes.pool import mean_to_avgpool
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Mean layer
    'mean_1': {'type': 'Pooling', 'kind': 'op', 'op': 'Mean', 'keep_dims': True},
    'mean_axis': {'value': None, 'shape': None, 'kind': 'data'},
    'mean_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # AvgPool layer
    'pool_1': {'type': 'Pooling', 'kind': 'op', 'op': 'Power', 'scale': None, 'shift': None, 'power': None},
    'pool_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Reshape layer
    'reshape_1': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class MeanToAvgPoolTests(unittest.TestCase):
    def _create_graph_with_mean(self, axis, keep_dims=True, mean_out_shape=np.array([1, 227, 227, 3])):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mean_1'),
                             ('mean_1', 'mean_1_data'),
                             ('mean_axis', 'mean_1'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'mean_1': {'shape': np.array([1, 227, 227, 3]), 'keep_dims': keep_dims},
                             'mean_axis': {'shape': np.array(axis.shape) if axis is not None else None,
                                           'value': np.array(axis) if axis is not None else None},
                             'mean_1_data': {'shape': mean_out_shape, 'is_output': True},
                             })
        del graph['mean_1']['mean_1_data'][0]['in']
        return graph

    def test_mean_to_avg_1(self):
        graph = self._create_graph_with_mean(axis=np.array([1, 2]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'pool_1'),
                                 ('pool_1', 'pool_1_data'),
                                 ],
                                {'pool_1': {'pool_method': 'avg', 'rounding_type': 'ceil', 'exclude_pad': 'true',
                                            'op': 'AvgPool', 'shape': np.array([1, 227, 227, 3])},
                                 'pool_1_data': {'is_output': True, 'shape': np.array([1, 227, 227, 3])}
                                 })

        mean_to_avgpool(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'mean_1_data', 'pool_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mean_to_avg_2(self):
        graph = self._create_graph_with_mean(axis=np.array([0]), keep_dims=False,
                                             mean_out_shape=np.array([227, 227, 3]))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'pool_1'),
                                 ('pool_1', 'pool_1_data'),
                                 ('pool_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data')
                                 ],
                                {'pool_1': {'pool_method': 'avg', 'rounding_type': 'ceil', 'exclude_pad': 'true',
                                            'op': 'AvgPool', 'shape': np.array([1, 227, 227, 3])},
                                 'pool_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'reshape_1_data': {'is_output': True, 'shape': np.array([227, 227, 3])},
                                 })

        mean_to_avgpool(graph)
        graph_clean_up(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'mean_1_data', 'reshape_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
