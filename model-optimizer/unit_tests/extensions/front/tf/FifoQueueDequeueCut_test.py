# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.front.tf.FifoQueueDequeueCut import FifoQueueDequeueCut
from mo.front.common.partial_infer.utils import shape_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_edge_attrs


class FifoQueueDequeueCutTest(unittest.TestCase):

    def test_one_output_v1(self):
        graph = build_graph_with_edge_attrs(
            {
                'queue_dequeue': {'kind': 'op', 'op': 'QueueDequeue', 'shapes': shape_array([[2, 2]]),
                                  'types': [np.int32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('queue_dequeue', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'type': np.int32},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        FifoQueueDequeueCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='sub')
        self.assertTrue(flag, msg)

    def test_one_output_v2(self):
        graph = build_graph_with_edge_attrs(
            {
                'queue_dequeue': {'kind': 'op', 'op': 'QueueDequeueV2', 'shapes': shape_array([[2, 2]]),
                                  'types': [np.int32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('queue_dequeue', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'type': np.int32},
                'sub': {'kind': 'op', 'op': 'Sub'},
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
            ]
        )

        FifoQueueDequeueCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='sub')
        self.assertTrue(flag, msg)

    def test_two_outputs_v1(self):
        graph = build_graph_with_edge_attrs(
            {
                'queue_dequeue': {'kind': 'op', 'op': 'QueueDequeue', 'shapes': [shape_array([2, 2]),
                                                                                shape_array([1, 1])],
                                 'types': [np.int32, np.float32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('queue_dequeue', 'sub', {'out': 0, 'in': 0}),
                ('queue_dequeue', 'add', {'out': 1, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'data_type': np.int32},
                'parameter_2': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([1, 1]), 'data_type': np.float32},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
                ('parameter_2', 'add', {'out': 0, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        FifoQueueDequeueCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='concat', check_op_attrs=True)
        self.assertTrue(flag, msg)

    def test_two_outputs_v2(self):
        graph = build_graph_with_edge_attrs(
            {
                'queue_dequeue': {'kind': 'op', 'op': 'QueueDequeueV2', 'shapes': [shape_array([2, 2]),
                                                                                shape_array([1, 1])],
                                 'types': [np.int32, np.float32]},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('queue_dequeue', 'sub', {'out': 0, 'in': 0}),
                ('queue_dequeue', 'add', {'out': 1, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        graph_ref = build_graph_with_edge_attrs(
            {
                'parameter_1': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([2, 2]), 'data_type': np.int32},
                'parameter_2': {'kind': 'op', 'op': 'Parameter', 'shape': shape_array([1, 1]), 'data_type': np.float32},
                'sub': {'kind': 'op', 'op': 'Sub'},
                'add': {'kind': 'op', 'op': 'Add'},
                'concat': {'kind': 'op', 'op': 'Concat'}
            },
            [
                ('parameter_1', 'sub', {'out': 0, 'in': 0}),
                ('parameter_2', 'add', {'out': 0, 'in': 0}),
                ('sub', 'concat', {'out': 0, 'in': 0}),
                ('add', 'concat', {'out': 0, 'in': 1})
            ]
        )

        FifoQueueDequeueCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='concat', check_op_attrs=True)
        self.assertTrue(flag, msg)
