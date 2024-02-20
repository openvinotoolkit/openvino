# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.middle.FusedBatchNormTraining import FusedBatchNormTraining
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.passes.eliminate import shape_inference
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

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

    'batchnorm': {'value': None, 'shape': int64_array([3, 10, 11, 5]), 'type': None, 'kind': 'op',
                  'op': 'FusedBatchNorm', 'is_training': True, 'eps': 1e-3},
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

    'start': {'kind': 'op', 'op': 'Const', 'value': int64_array(1)},
    'start_data': {'value': None, 'shape': None, 'kind': 'data', 'value': int64_array(1)},
    'stop': {'kind': 'op', 'op': 'Const', 'value': int64_array(3)},
    'stop_data': {'value': None, 'shape': None, 'kind': 'data', 'value': int64_array(3)},
    'step': {'kind': 'op', 'op': 'Const', 'value': int64_array(1)},
    'step_data': {'value': None, 'shape': None, 'kind': 'data', 'value': int64_array(1)},
    'mvn_axes': {'kind': 'op', 'op': 'Range'},
    'mvn_axes_data': {'value': None, 'shape': None, 'kind': 'data'},

    'mvn': {'type': 'MVN', 'value': None, 'kind': 'op', 'op': 'MVN', 'eps': 1e-3, 'eps_mode': 'inside_sqrt'},
    'mvn_data': {'value': None, 'shape': None, 'kind': 'data'},

    'reshape_1': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([1, -1, 0, 0])},
    'reshape_1_const_data': {'kind': 'data', 'value': None, 'shape': None},
}


class TestFusedBatchNormTrainingTest():
    @pytest.mark.parametrize("op",[
        'FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3',
    ])
    def test_transformation(self, op: str):
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
        graph.nodes['batchnorm']['op'] = op
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
                                 ('start', 'start_data'),
                                 ('start_data', 'mvn_axes'),
                                 ('stop', 'stop_data'),
                                 ('stop_data', 'mvn_axes'),
                                 ('step', 'step_data'),
                                 ('step_data', 'mvn_axes'),
                                 ('mvn_axes', 'mvn_axes_data'),
                                 ('mvn_axes_data', 'mvn'),
                                 ('placeholder_data', 'shapeof', {'in': 0}),
                                 ('shapeof', 'shapeof_data'),
                                 ('shapeof_data', 'reshape_to_orig', {'in': 1}),
                                 ('reshape_to_orig', 'reshape_to_orig_data'),
                                 ('reshape_to_orig_data', 'batchnorm', {'in': 0}),

                                 ('batchnorm', 'batchnorm_data'),
                                 ('batchnorm_data', 'result'),
                                 ],
                                {'batchnorm': {'is_training': False},

                                 }, nodes_with_edges_only=True)
        FusedBatchNormTraining().find_and_replace_pattern(graph)
        shape_inference(graph)

        graph_ref.nodes['batchnorm']['op'] = op

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp

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
                            {'batchnorm': {'is_training': False}}, nodes_with_edges_only=True)
        graph_ref = graph.copy()

        FusedBatchNormTraining().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        assert flag, resp
