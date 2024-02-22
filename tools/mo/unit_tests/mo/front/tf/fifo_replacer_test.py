# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.front.tf.fifo_replacer import FIFOQueue, FIFOQueueDequeueCut
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_edge_attrs


def create_fifo_queue_graph(batch_size_shape: np.ndarray):
    nodes = {
        'placeholder': {'op': 'Parameter', 'data_type': np.int32, 'kind': 'op', 'shape': batch_size_shape},
        'batch_join/fifo_queue': {'op': 'FIFOQueueV2', 'name': 'batch_join/fifo_queue',
                                  'shapes': np.array([[1, 2, 3]]), 'types': np.array([np.float32]), 'kind': 'op'},
        'batch_join': {'op': 'QueueDequeueUpToV2', 'kind': 'op'},
        'image_batch': {'op': 'Identity', 'data_type': np.float32, 'kind': 'op'},
        'label_batch': {'op': 'Identity', 'kind': 'op'},
        'label_batch_op_output': {'op': 'Result', 'kind': 'op'},
    }
    edges = [
        ('placeholder', 'batch_join', {'out': 0, 'in': 0}),
        ('batch_join/fifo_queue', 'batch_join', {'out': 0, 'in': 1}),
        ('batch_join', 'image_batch', {'out': 0, 'in': 0}),
        ('batch_join', 'label_batch', {'out': 1, 'in': 0}),
        ('label_batch', 'label_batch_op_output', {'out': 0, 'in': 0})
    ]
    graph = build_graph_with_edge_attrs(nodes, edges)
    return graph


class TestFIFOQueueReplacement(unittest.TestCase):
    def test_fifo_with_label_batch(self):
        graph = create_fifo_queue_graph(shape_array([1]))
        tested_class = FIFOQueue()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        self.assertEqual(2, len(after_pattern))
        try:
            new_ph_dict = graph.node[[u for u, v in graph.in_edges('image_batch')][0]]
        except Exception as e:
            self.fail("Can't get new placeholder. Broken edge. Additional information: {}".format(e))
        self.assertEqual(new_ph_dict['name'], 'batch_join/fifo_queue')
        self.assertTrue(np.array_equal(new_ph_dict['shape'], [1, 2, 3]))

    def test_fifo_with_undefined_batch_size(self):
        graph = create_fifo_queue_graph(None)
        tested_class = FIFOQueue()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        self.assertEqual(2, len(after_pattern))
        try:
            new_ph_dict = graph.node[[u for u, v in graph.in_edges('image_batch')][0]]
        except Exception as e:
            self.fail("Can't get new placeholder. Broken edge. Additional information: {}".format(e))
        self.assertEqual(new_ph_dict['name'], 'batch_join/fifo_queue')
        self.assertTrue(np.array_equal(new_ph_dict['shape'], [1, 2, 3]))

    def test_fifo_with_out_label_batch(self):
        nodes_no_label = {
            'placeholder': {'op': 'Parameter', 'data_type': np.int32, 'kind': 'op', 'shape': np.array(0)},
            'batch_join/fifo_queue': {'op': 'FIFOQueueV2', 'name': 'batch_join/fifo_queue',
                                      'shapes': np.array([[1, 2, 3]]), 'types': np.array([np.float32]), 'kind': 'op'},
            'batch_join': {'op': 'QueueDequeueUpToV2', 'kind': 'op'},
            'image_batch': {'op': 'Identity', 'data_type': np.float32, 'kind': 'op'},
        }
        edges_no_label = [
            ('placeholder', 'batch_join', {'out': 0, 'in': 0}),
            ('batch_join/fifo_queue', 'batch_join', {'out': 0, 'in': 1}),
            ('batch_join', 'image_batch', {'out': 0, 'in': 0})
        ]

        graph = build_graph_with_edge_attrs(nodes_no_label, edges_no_label)
        tested_class = FIFOQueue()
        tested_class.find_and_replace_pattern(graph=graph)
        after_pattern = graph.nodes()
        self.assertEqual(2, len(after_pattern))
        try:
            new_ph_dict = graph.node[[u for u, v in graph.in_edges('image_batch')][0]]
        except Exception as e:
            self.fail("Can't get new placeholder. Broken edge. Additional information: {}".format(e))
        self.assertEqual(new_ph_dict['name'], 'batch_join/fifo_queue')
        self.assertTrue(np.array_equal(new_ph_dict['shape'], np.array([1, 2, 3])))


class FIFOQueueDequeueCutTest(unittest.TestCase):
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

        FIFOQueueDequeueCut().find_and_replace_pattern(graph)

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

        FIFOQueueDequeueCut().find_and_replace_pattern(graph)

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

        FIFOQueueDequeueCut().find_and_replace_pattern(graph)

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

        FIFOQueueDequeueCut().find_and_replace_pattern(graph)

        flag, msg = compare_graphs(graph, graph_ref, last_node='concat', check_op_attrs=True)
        self.assertTrue(flag, msg)
