"""
 Copyright (c) 2018 Intel Corporation

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

import logging as log

import numpy as np

from mo.front.extractor import update_attrs
from mo.ops.op import PermuteAttrs


def tf_transpose_infer(node):
    if len(node.in_nodes()) != 2:
        log.error("Transpose should take 2 inputs")
        return

    node_inp, node_order = (node.in_node(0), node.in_node(1))
    order = node_order.value
    in_shape = np.array(node_inp.shape)
    node.graph.remove_edge(node_order.node, node.node)
    node.order = np.array(order)
    node.out_node().shape = in_shape[order]
    if node_inp.has_valid('value'):
        node.out_node().value = np.transpose(node_inp.value, axes=order)

    PermuteAttrs.create_permute_attrs(node, attrs=[('order','input:0')])


def transpose_infer(node):
    if node.order is None and (not node.has_valid('reverse_order') or (node.has_valid('reverse_order') and node.reverse_order == False)):
        log.error('Cannot infer {} because order is None'.format(node.soft_get('name')))
        return

    if node.has_valid('reverse_order') and node.reverse_order and node.has_valid('order'):
        log.error('Cannot infer {} due to both order and reverse_order was set'.format(node.soft_get('name')))
        return

    input_shape = node.in_node(0).shape

    if node.has_valid('reverse_order') and node.reverse_order:
        node.order = np.arange(len(input_shape))[::-1] # Reverse order

    output_shape = np.array([input_shape[i] for i in node.order], dtype=np.int64)
    node.out_node(0).shape = output_shape
    if node.in_node().has_valid('value'):
        node.out_node().value = np.transpose(node.in_node().value, axes=node.order)
    PermuteAttrs.create_permute_attrs(node, attrs=[('order', 'input:0')])
