# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.mxnet.extractors.slice_axis import mxnet_slice_axis_infer
from openvino.tools.mo.front.mxnet.extractors.slice_axis import slice_axis_ext
from openvino.tools.mo.front.mxnet.extractors.utils import AttrDictionary
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestMXNetSliceAxisExtractorOp(unittest.TestCase):
    def test_extract_slice_axis_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'slice_axis_node': {'type': 'sigmoid', 'kind': 'op', 'op': 'slice_axis', },
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'slice_axis_node'),
                ('slice_axis_node', 'node_3'),
            ],
            {
                'slice_axis_node': {'symbol_dict': {'attrs': {'axis': 0, 'begin': 10, 'end': 25}}},
            })

        exp_attrs = {
            'op': 'Crop',
            'axis': 0,
            'offset': 10,
            'dim': 25
        }

        slice_axis_node = Node(graph, 'slice_axis_node')
        res = slice_axis_ext(AttrDictionary(slice_axis_node['symbol_dict']['attrs']))

        for key in exp_attrs.keys():
            self.assertEqual(res[key], exp_attrs[key])


class TestMXNetSliceAxisInfer(unittest.TestCase):
    def test_slice_axis_infer_layer(self):
        graph = build_graph(
            {'node_1': {'name': 'data', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'slice_axis_node': {'name': 'slice_axis_node', 'type': 'sigmoid', 'value': None,
                                 'kind': 'op', 'op': 'slice_axis', },
             'node_3': {'name': 'node_3', 'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'slice_axis_node'),
                ('slice_axis_node', 'node_3'),
            ],
            {
                'node_1': {'shape': np.array([1, 1024, 19, 19])},
                'slice_axis_node': {'axis': 1, 'offset': 10, 'dim': 25},
            })

        slice_axis_node = Node(graph, 'slice_axis_node')
        mxnet_slice_axis_infer(slice_axis_node)
        res_shape = [1, 15, 19, 19]
        for i in range(0, len(graph.node['node_3']['shape'])):
            self.assertEqual(graph.node['node_3']['shape'][i], res_shape[i])
