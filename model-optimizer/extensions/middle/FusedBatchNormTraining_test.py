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
from generator import generator

from extensions.middle.FusedBatchNormTraining import FusedBatchNormTraining
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.middle.passes.eliminate_test import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'type': 'Parameter', 'kind': 'op',
                    'op': 'Parameter'},
    'placeholder_data': {'shape': int64_array([3, 10, 11, 5]), 'value': None, 'kind': 'data'},

    'scale': {'value': np.array([2, 3.5, 4.5, 5.1, 2.6], dtype=np.float32), 'shape': int64_array([5]), 'kind': 'op',
              'op': 'Const'},
    'scale_data': {'value': np.array([2, 3.5, 4.5, 5.1, 2.6], dtype=np.float32), 'shape': int64_array([5]),
                   'kind': 'data'},

    'offset': {'value': np.array([1, 2.5, 3.5, 4.1, 5.6], dtype=np.float32), 'shape': int64_array([5]), 'kind': 'op',
               'op': 'Const'},
    'offset_data': {'value': np.array([1, 2.5, 3.5, 4.1, 5.6], dtype=np.float32), 'shape': int64_array([5]),
                    'kind': 'data'},

    'mean': {'value': None, 'shape': int64_array([]), 'kind': 'op', 'op': 'Const'},
    'mean_data': {'value': None, 'shape': int64_array([]), 'kind': 'data'},

    'variance': {'value': None, 'shape': int64_array([]), 'kind': 'op', 'op': 'Const'},
    'variance_data': {'value': None, 'shape': int64_array([]), 'kind': 'data'},

    'batchnorm_train': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'type': None, 'kind': 'op',
                        'op': 'BatchNormTraining', 'eps': 1e-3},
    'batchnorm_train_data': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'kind': 'data'},

    'batchnorm': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'type': None, 'kind': 'op',
                  'op': 'BatchNormInference', 'eps': 1e-3},
    'batchnorm_data': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'kind': 'data'},

    'result': {'kind': 'op', 'op': 'Result'},

    # nodes after transformation
    'bn_mean': {'value': np.zeros([5]), 'shape': int64_array([5]), 'kind': 'op', 'op': 'Const'},
    'bn_mean_data': {'value': np.zeros([5]), 'shape': int64_array([5]), 'kind': 'data'},

    'bn_variance': {'value': np.ones([5]), 'shape': int64_array([5]), 'kind': 'op', 'op': 'Const'},
    'bn_variance_data': {'value': np.ones([5]), 'shape': int64_array([5]), 'kind': 'data'},

    'shapeof': {'type': 'ShapeOf', 'value': None, 'kind': 'op', 'op': 'ShapeOf'},
    'shapeof_data': {'value': int64_array([3, 10, 11, 5]), 'shape': int64_array([4]), 'kind': 'data'},

    'reshape_to_orig': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_to_orig_data': {'value': None, 'shape': None, 'kind': 'data'},

    'mvn': {'type': 'MVN', 'value': None, 'kind': 'op', 'op': 'MVN', 'eps': 1e-3},
    'mvn_data': {'value': None, 'shape': None, 'kind': 'data'},

    'reshape_1': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1, 0, 0])},
    'reshape_1_const_data': {'kind': 'data', 'value': None, 'shape': None},
}


@generator
class FusedBatchNormTrainingTest(unittest.TestCase):
    def test_transformation(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data', {}),
                             ('scale', 'scale_data'),
                             ('offset', 'offset_data'),
                             ('mean', 'mean_data'),
                             ('variance', 'variance_data'),
                             ('placeholder_data', 'batchnorm_train', {'in': 0}),
                             ('scale_data', 'batchnorm_train', {'in': 1}),
                             ('offset_data', 'batchnorm_train', {'in': 2}),
                             ('mean_data', 'batchnorm_train', {'in': 3}),
                             ('variance_data', 'batchnorm_train', {'in': 4}),
                             ('batchnorm_train', 'batchnorm_train_data'),
                             ('batchnorm_train_data', 'result'),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'placeholder_data', {}),
                                 ('scale', 'scale_data'),
                                 ('offset', 'offset_data'),
                                 ('bn_mean', 'bn_mean_data'),
                                 ('bn_variance', 'bn_variance_data'),
                                 ('scale_data', 'batchnorm', {'in': 1}),
                                 ('offset_data', 'batchnorm', {'in': 2}),
                                 ('bn_mean_data', 'batchnorm', {'in': 3}),
                                 ('bn_variance_data', 'batchnorm', {'in': 4}),

                                 ('placeholder_data', 'reshape_1', {'in': 0}),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1', {'in': 1}),
                                 ('reshape_1', 'reshape_1_data', {}),
                                 ('reshape_1_data', 'mvn', {'in': 0}),
                                 ('mvn', 'mvn_data'),
                                 ('mvn_data', 'reshape_to_orig', {'in': 0}),
                                 ('placeholder_data', 'shapeof', {'in': 0}),
                                 ('shapeof', 'shapeof_data'),
                                 ('shapeof_data', 'reshape_to_orig', {'in': 1}),
                                 ('reshape_to_orig', 'reshape_to_orig_data'),
                                 ('reshape_to_orig_data', 'batchnorm', {'in': 0}),

                                 ('batchnorm', 'batchnorm_data'),
                                 ('batchnorm_data', 'result'),
                                 ],
                                {}, nodes_with_edges_only=True)
        FusedBatchNormTraining().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_non_training(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data', {}),
                             ('scale', 'scale_data'),
                             ('offset', 'offset_data'),
                             ('mean', 'mean_data'),
                             ('variance', 'variance_data'),
                             ('placeholder_data', 'batchnorm', {'in': 0}),
                             ('scale_data', 'batchnorm', {'in': 1}),
                             ('offset_data', 'batchnorm', {'in': 2}),
                             ('mean_data', 'batchnorm', {'in': 3}),
                             ('variance_data', 'batchnorm', {'in': 4}),
                             ('batchnorm', 'batchnorm_data'),
                             ('batchnorm_data', 'result'),
                             ],
                            {}, nodes_with_edges_only=True)
        graph_ref = graph.copy()

        FusedBatchNormTraining().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
