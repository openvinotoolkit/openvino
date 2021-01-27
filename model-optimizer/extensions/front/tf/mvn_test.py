"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.front.tf.mvn import MVNReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'ReduceMean'},
    'stop_grad': {'value': None, 'shape': None, 'kind': 'op', 'op': 'StopGradient'},
    'sqdiff': {'value': None, 'shape': None, 'kind': 'op', 'op': 'SquaredDifference'},
    'variance': {'value': None, 'shape': None, 'kind': 'op', 'op': 'ReduceMean'},
    'squeeze_mean': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Squeeze'},
    'squeeze_variance': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Squeeze'},
    'fbn': {'value': None, 'shape': None, 'kind': 'op', 'op': 'BatchNormInference'},
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'conv': {'type': 'Convolution', 'kind': 'op', 'op': 'Convolution'},
    'gamma_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'beta_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'mean_reduction_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'variance_reduction_const': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Const'},
    'mvn': {'value': None, 'shape': None, 'kind': 'op', 'op': 'MVN'},
    'conv_rank': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Rank'},
    'one_const': {'value': np.int32(1), 'shape': None, 'kind': 'op', 'op': 'Const'},
    'range_limit': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Sub'},
    'range': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Range'},
    'new_beta': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'new_gamma': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Unsqueeze'},
    'mvn_mul_gamma': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Mul'},
    'add_beta': {'value': None, 'shape': None, 'kind': 'op', 'op': 'Add'},
}


class MVNReplaceTests(unittest.TestCase):

    def test_nchw_layout(self):
        pattern_matcher = MVNReplacer()
        graph = build_graph(nodes_attributes,
                            [('conv', 'fbn'),
                             ('conv', 'mean', {'in': 0}),
                             ('mean_reduction_const', 'mean', {'in': 1}),
                             ('variance_reduction_const', 'variance', {'in': 1}),
                             ('conv', 'sqdiff'),
                             ('gamma_const', 'fbn', {'in': 1}),
                             ('beta_const', 'fbn', {'in': 2}),
                             ('mean', 'stop_grad', {'in': 0}),
                             ('stop_grad', 'sqdiff', {'in': 1}),
                             ('sqdiff', 'variance', {'in': 0}),
                             ('mean', 'squeeze_mean', {'in': 0}),
                             ('variance', 'squeeze_variance', {'in': 0}),
                             ('squeeze_mean', 'fbn', {'in': 3}),
                             ('squeeze_variance', 'fbn', {'in': 4}),
                             ('fbn', 'concat')],
                            {'fbn': {'eps': 1.2, 'data_format': b'NCHW'},
                             'mean_reduction_const': {'value': np.array([2, 3])},
                             'variance_reduction_const': {'value': np.array([2, 3])},
                             'gamma_const': {'value': np.float32(1.)},
                             'beta_const': {'value': np.float32(1.)}, },
                            nodes_with_edges_only=True)
        graph.stage = 'front'
        graph.graph['layout'] = b'NHWC'
        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes, [('conv', 'mvn'),
                                                   ('conv', 'mean', {'in': 0}),
                                                   ('conv', 'sqdiff'),
                                                   ('conv', 'conv_rank'),
                                                   ('mean_reduction_const', 'mean', {'out': 0, 'in': 1}),
                                                   ('variance_reduction_const', 'variance', {'out': 0, 'in': 1}),
                                                   ('mean', 'stop_grad', {'in': 0}),
                                                   ('stop_grad', 'sqdiff', {'in': 1}),
                                                   ('sqdiff', 'variance', {'in': 0}),
                                                   ('mean', 'squeeze_mean', {'in': 0}),
                                                   ('variance', 'squeeze_variance', {'in': 0}),
                                                   ('mean_reduction_const', 'mvn', {'out': 0}),
                                                   ('variance_reduction_const', 'mvn', {'out': 0}),
                                                   ('conv_rank', 'range_limit'),
                                                   ('one_const', 'range_limit', {'out': 0}),
                                                   ('one_const', 'range', {'out': 0}),
                                                   ('range_limit', 'range'),
                                                   ('one_const', 'range', {'out': 0}),
                                                   ('beta_const', 'new_beta'),
                                                   ('range', 'new_beta', {'out': 0}),
                                                   ('gamma_const', 'new_gamma'),
                                                   ('range', 'new_gamma', {'out': 0}),
                                                   ('mvn', 'mvn_mul_gamma'),
                                                   ('new_gamma', 'mvn_mul_gamma'),
                                                   ('mvn_mul_gamma', 'add_beta'),
                                                   ('new_beta', 'add_beta'),
                                                   ('add_beta', 'concat')], {}, nodes_with_edges_only=True)

        graph_ref.stage = 'front'
        graph_ref.graph['layout'] = b'NHWC'

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)
