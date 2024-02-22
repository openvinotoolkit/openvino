# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array

import numpy as np
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.roialign import ROIAlign
from unit_tests.utils.graph import build_graph


class TestROIAlignOps(unittest.TestCase):
    node_attrs = {
        # input 1
        "1_input": {"kind": "op", "type": "Parameter", "value": None},
        "input_data": {"shape": None, "kind": "data", "value": None},
        #input 2
        "2_rois": {"kind": "op", "type": "Parameter","value": None},
        "rois_data": {"shape": None,"kind": "data", "value": None},
        # input 3
        "3_indices": {"kind": "op","type": "Parameter"},
        "indices_data": {"shape": None, "kind": "data", "value": None},
        # ROIAlign
        "node": {
            "kind": "op",
            "type": "ROIAlign",
            "pooled_h": None,
            "pooled_w": None,
            "mode": None,
            "sampling_ratio": 2,
            "spatial_scale": 16,
            "aligned_mode": None,
        },
        "node_data": {"shape": None, "kind": "data", "value": None},
        # output
        "result": {"kind": "op","type": "Result"},
    }

    def test_roialignv1(self):
        graph = build_graph(
            self.node_attrs,
            [
                ("1_input", "input_data"),
                ("input_data", "node", {"in": 0}),
                ("2_rois", "rois_data"),
                ("rois_data", "node", {"in": 1}),
                ("3_indices", "indices_data"),
                ("indices_data", "node", {"in": 2}),
                ("node", "node_data"),
                ("node_data", "result"),
            ],
            {
                'input_data': {'shape': int64_array([1, 256, 200, 272])},
                'rois_data': {'shape': int64_array([1000, 4])},
                'indices_data': {'shape': int64_array([1000])},
                'node': {'mode': 'max', 'pooled_h': 7, 'pooled_w': 7, 'aligned_mode': 'asymmetric', 'version': 'opset9'},
            }
        )
        graph.graph["layout"] = "NCHW"
        node = Node(graph, "node")
        ROIAlign.infer(node)
        self.assertListEqual(list([1000, 256, 7, 7]), graph.node['node_data']['shape'].data.tolist())

    def test_roialignv2(self):
        graph = build_graph(
            self.node_attrs,
            [
                ("1_input", "input_data"),
                ("input_data", "node", {"in": 0}),
                ("2_rois", "rois_data"),
                ("rois_data", "node", {"in": 1}),
                ("3_indices", "indices_data"),
                ("indices_data", "node", {"in": 2}),
                ("node", "node_data"),
                ("node_data", "result"),
            ],
            {
                'input_data': {'shape': int64_array([7, 256, 200, 200])},
                'rois_data': {'shape': int64_array([300, 4])},
                'indices_data': {'shape': int64_array([300])},
                'node': {'mode': 'max', 'pooled_h': 5, 'pooled_w': 6, 'aligned_mode': 'half_pixel_for_nn', 'version':'opset9'},
            }
        )
        graph.graph["layout"] = "NCHW"
        node = Node(graph, "node")

        ROIAlign.infer(node)
        self.assertListEqual(list([300, 256, 5, 6]), graph.node['node_data']['shape'].data.tolist())

    def test_roialignv3(self):
        graph = build_graph(
            self.node_attrs,
            [
                ("1_input", "input_data"),
                ("input_data", "node", {"in": 0}),
                ("2_rois", "rois_data"),
                ("rois_data", "node", {"in": 1}),
                ("3_indices", "indices_data"),
                ("indices_data", "node", {"in": 2}),
                ("node", "node_data"),
                ("node_data", "result"),
            ],
            {
                'input_data': {'shape': int64_array([2, 3, 5, 5])},
                'rois_data': {'shape': int64_array([7, 4])},
                'indices_data': {'shape': int64_array([7])},
                'node': {'mode': 'max', 'pooled_h': 2, 'pooled_w': 2, 'aligned_mode': 'half_pixel', 'version': 'opset9'},
            }
        )
        graph.graph["layout"] = "NCHW"
        node = Node(graph, "node")

        ROIAlign.infer(node)
        self.assertListEqual(list([7, 3, 2, 2]), graph.node['node_data']['shape'].data.tolist())


    def test_roialign_wrong_aligned_mode(self):
        graph = build_graph(
            self.node_attrs,
            [
                ("1_input", "input_data"),
                ("input_data", "node", {"in": 0}),
                ("2_rois", "rois_data"),
                ("rois_data", "node", {"in": 1}),
                ("3_indices", "indices_data"),
                ("indices_data", "node", {"in": 2}),
                ("node", "node_data"),
                ("node_data", "result"),
            ],
            {
                'input_data': {'shape': int64_array([2, 3, 5, 5])},
                'rois_data': {'shape': int64_array([7, 4])},
                'indices_data': {'shape': int64_array([7])},
                'node': {'mode': 'max', 'pooled_h': 2, 'pooled_w': 2, 'aligned_mode': 'full_pixel', 'version': 'opset9'},
            }
        )
        graph.graph["layout"] = "NCHW"
        node = Node(graph, "node")
        self.assertRaises(AssertionError, ROIAlign.infer, node)
