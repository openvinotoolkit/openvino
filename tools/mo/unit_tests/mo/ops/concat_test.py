# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.ops.concat import Concat
from unit_tests.utils.graph import build_graph


class TestConcatOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([227, 227, 227, 227])
        },
        'concat_node': {
        },
        'node_3': {
            'kind': 'data'
        }
    }

    def test_concat_op(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'concat_node'),
                                ('concat_node', 'node_3')
                            ])
        concat_node = Concat(graph, self.nodes_attributes['concat_node']).add_node()
        self.assertEqual(concat_node.type, 'Concat')
        self.assertEqual(concat_node.op, 'Concat')
        self.assertEqual(concat_node.infer, concat_infer)
