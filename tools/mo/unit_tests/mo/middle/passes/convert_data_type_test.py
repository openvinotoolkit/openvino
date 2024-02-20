# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.passes.convert_data_type import convert_blobs, SUPPORTED_DATA_TYPES
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import build_graph

nodes_attributes = {'data_node': {'kind': 'data', 'value': None, 'shape': int64_array([5])},
                    'op_node': { 'kind': 'op', 'op': 'Result'}}


class TestConvertBlob(UnitTestWithMockedTelemetry):
    def test_convert_blob_to_fp32_from_fp64(self):
        graph = build_graph(nodes_attributes,
                            [('data_node', 'op_node', {'bin': 1})],
                            {'data_node': {'value': np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)}})

        convert_blobs(graph, "FP32")
        result_value = graph.node['data_node']['value']
        self.assertTrue(result_value.dtype == np.float32)
        self.assertListEqual(list(result_value), [4, 3, 2, 1])

    def test_convert_blob_to_fp16_from_fp64(self):
        graph = build_graph(nodes_attributes,
                            [('data_node', 'op_node', {'bin': 1})],
                            {'data_node': {'value': np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)}})

        convert_blobs(graph, "FP16")
        result_value = graph.node['data_node']['value']
        self.assertTrue(result_value.dtype == np.float16)
        self.assertListEqual(list(result_value), [4, 3, 2, 1])

    def test_convert_blob_to_fp16_from_fp64_overflow(self):
        graph = build_graph(nodes_attributes,
                            [('data_node', 'op_node', {'bin': 1})],
                            {'data_node': {'value': np.array([4.0, 3.0, 2.0, 1e10], dtype=np.float64)}})

        convert_blobs(graph, "FP16")
        result_value = graph.node['data_node']['value']
        self.assertTrue(result_value.dtype == np.float16)
        self.assertListEqual(list(result_value), [4, 3, 2, np.inf])

    def test_convert_blob_to_int32_with_force_precision(self):
        graph = build_graph(nodes_attributes,
                            [('data_node', 'op_node', {'bin': 1})],
                            {'data_node': {'value': np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)}})

        convert_blobs(graph, "I32")
        result_value = graph.node['data_node']['value']
        self.assertTrue(result_value.dtype == np.int32)
        self.assertListEqual(list(result_value), [4, 3, 2, 1])

    def test_convert_blob_to_int32_with_force_precision_error(self):
        graph = build_graph(nodes_attributes,
                            [('data_node', 'op_node', {'bin': 1})],
                            {'data_node': {'value': np.array([4.0, 3.0, 2.0, 1.1], dtype=np.float64)}})

        with self.assertRaisesRegex(Error, '.*results in rounding.*'):
            convert_blobs(graph, "I32")


class TestUI8(unittest.TestCase):
    def test_supported_data_types_uint8_once(self):
        i = 0
        for data_type_str, values in SUPPORTED_DATA_TYPES.items():
            np_dt, precision, element_type = values
            if np_dt == np.uint8:
                i += 1

        self.assertEqual(i, 1, 'uint8 data type should be mentioned in SUPPORTED_DATA_TYPES only once, {} entries '
                               'found'.format(i))
