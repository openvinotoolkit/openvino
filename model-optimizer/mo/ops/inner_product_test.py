"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.common.partial_infer.inner_product import caffe_inner_product
from mo.ops.inner_product import InnerProduct
from mo.utils.unittest.graph import build_graph


class TestInnerProductOp(unittest.TestCase):
    # There are tests for InnerProduct.infer in mo/front/common/partial_infer/inner_product_test.py
    nodes_attributes = {
        'node_1': {
            'shape': np.array([227, 5, 2, 1])
        },
        'fc_node': {
        },
        'node_3': {
            'kind': 'data'
        }
    }

    def test_concat_op(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'fc_node'),
                                ('fc_node', 'node_3')
                            ])
        fc_node = InnerProduct(graph, self.nodes_attributes['fc_node']).add_node()
        self.assertEqual(fc_node.type, 'MatMul')
        self.assertEqual(fc_node.op, 'MatMul')
        self.assertEqual(fc_node.infer, caffe_inner_product)
