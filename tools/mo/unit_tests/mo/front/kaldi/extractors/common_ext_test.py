# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.graph.graph import Node, Graph
from unit_tests.mo.front.kaldi.loader.utils_test import TestKaldiUtilsLoading
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import build_graph


class KaldiFrontExtractorTest(UnitTestWithMockedTelemetry):
    graph = Graph()
    nodes_attributes = {}
    test_node = None

    @classmethod
    def setUp(cls):
        super().setUp(cls)
        cls.nodes_attributes = {
            'input_data_node': {
                'name': 'input_data_node',
                'kind': 'data',
                'shape': np.array([1, 32, 1, 40], dtype=np.int64),
            },
            'weights': {
                'name': 'weights',
                'kind': 'data',
                'shape': np.array([10, 32, 1, 8], dtype=np.int64),
                'value': np.zeros((10, 32, 1, 8)),
                'dim_attrs': ['spatial_dims', 'channel_dims', 'batch_dims', 'axis'],
            },
            'test_node': {
                'name': 'test_node',
                'kind': 'op'
            },
            'output_data_node': {
                'name': 'output_data_node',
                'kind': 'data',
                'shape': None
            }
        }
        cls.create_graph()
        cls.test_node = Node(cls.graph, 'test_node')
        cls.graph.add_node(cls.test_node.id, type='test_node')
        cls.register_op()
        cls.create_pb_for_test_node()

    @staticmethod
    def register_op():
        raise NotImplementedError('Please, implement register_op')

    @classmethod
    def create_graph(cls):
        cls.graph = build_graph(cls.nodes_attributes, [
            ('input_data_node', 'test_node'),
            ('test_node', 'output_data_node')
        ], nodes_with_edges_only=True)

    @classmethod
    def create_pb_for_test_node(cls):
        pass

    @staticmethod
    def generate_learn_info():
        pb = KaldiFrontExtractorTest.write_tag_with_value('<LearnRateCoef>', 0)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<BiasLearnRateCoef>', 1)
        pb += KaldiFrontExtractorTest.write_tag_with_value('<MaxNorm>', 2)
        return pb

    @staticmethod
    def generate_matrix(shape):
        pb = KaldiFrontExtractorTest.write_tag_with_value('FM', shape[0])
        pb += KaldiFrontExtractorTest.write_int_value(shape[1])
        pb += KaldiFrontExtractorTest.generate_blob(np.prod(shape))
        return pb

    @staticmethod
    def generate_vector(size: int) -> bytes:
        pb = KaldiFrontExtractorTest.write_tag_with_value('FV', size)
        pb += KaldiFrontExtractorTest.generate_blob(size)
        return pb

    @staticmethod
    def generate_blob(size: int) -> bytes:
        pb = b''
        for i in range(size):
            pb += TestKaldiUtilsLoading.pack_value(i, TestKaldiUtilsLoading.float32_fmt)
        return pb

    @staticmethod
    def write_tag_with_value(tag: str, value, value_type=np.int32) -> bytes:
        pb = bytes(tag + ' ', 'ascii')
        if value_type == np.int32:
            return pb + KaldiFrontExtractorTest.write_int_value(value)
        elif value_type == np.float32:
            return pb + KaldiFrontExtractorTest.write_float_value(value)
        else:
            return pb + KaldiFrontExtractorTest.write_str_value(value)

    @staticmethod
    def write_int_value(value) -> bytes:
        pb = TestKaldiUtilsLoading.pack_value(4, 'B')
        pb += TestKaldiUtilsLoading.pack_value(value, TestKaldiUtilsLoading.uint32_fmt)
        return pb

    @staticmethod
    def write_float_value(value) -> bytes:
        pb = TestKaldiUtilsLoading.pack_value(4, 'B')
        pb += TestKaldiUtilsLoading.pack_value(value, TestKaldiUtilsLoading.float32_fmt)
        return pb

    @staticmethod
    def write_str_value(value) -> bytes:
        pb = bytes(value, 'ascii')
        return pb

    def compare_node_attrs(self, exp_res):
        node = self.test_node
        for key in exp_res.keys():
            if type(node[key]) in [list, np.ndarray]:
                self.assertTrue(np.array_equal(np.array(node[key]), np.array(exp_res[key])))
            else:
                self.assertEqual(node[key], exp_res[key])
