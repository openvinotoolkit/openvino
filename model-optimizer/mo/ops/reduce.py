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

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.ops.op import Op


class Reduce(Op):
    enabled = False

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'op': 'Reduce',
            'reduce_type': None,
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        input_node = node.in_node()
        input_shape = np.array(node.in_node().shape, dtype=np.int64)
        output_node = node.out_node()

        if not node.has_valid('reduce_type'):
            log.error('Reduce type for node {} not specified!'.format(node.id))
            return

        reduce_type = node.reduce_type

        if input_node.has_valid('value'):
            if reduce_type.lower() == 'mean':
                # Value and Shape propagation for constant path
                output_node.value = np.mean(input_node.value, axis=tuple(node.axis), keepdims=node.keep_dims)
                output_node.shape = output_node.value.shape
            else:
                log.error('Reduce type {} is not supported for node {}'.format(reduce_type, node.id))
                return
        else:
            # In case if axis is None it means that reduction comes along each dimension
            if node.axis is None:
                node.axis = np.array(range(len(input_shape)))

            used_dims = np.zeros(len(input_shape), dtype=np.bool)
            output_shape = input_shape

            for dim in node.axis:
                used_dims[dim] = True
                output_shape[dim] = 1

            # In case if keep dims == False, we should remove all 1 dims that was used in reduction
            if not node.keep_dims:
                output_shape = output_shape[np.invert(used_dims)]

            output_node.shape = output_shape

