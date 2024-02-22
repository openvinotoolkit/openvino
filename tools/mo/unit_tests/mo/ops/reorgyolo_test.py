# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.reorgyolo import ReorgYoloOp
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'reorg': {'type': 'ReorgYolo', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


def calculate_reorgyolo_output(input, stride):
    output = np.full_like(input, -1, dtype=np.int64)
    output[0] = input[0]
    output[1] = input[1] * stride ** 2
    output[2] = np.round(input[2] / stride)
    output[3] = np.round(input[3] / stride)
    return output


class TestReorgYOLO(unittest.TestCase):
    def test_reorgyolo_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'reorg'),
                             ('reorg', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'value': None},
                             'reorg': {'stride': 2,
                                       **layout_attrs()}
                             })

        reorg_node = Node(graph, 'reorg')
        ReorgYoloOp.reorgyolo_infer(reorg_node)
        exp_shape = calculate_reorgyolo_output(np.array([1, 3, 227, 227]), 2)
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
