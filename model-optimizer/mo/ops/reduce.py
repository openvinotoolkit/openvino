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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error


class Reduce(Op):
    enabled = False

    reduce_method_map = {
        'max': np.max,
        'mean': np.mean,
        'sum': np.sum,
    }

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'op': 'Reduce',
            'reduce_type': None,
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        if len(node.in_nodes()) == 2:
            reduction_indices_data = node.in_node(1)
            if reduction_indices_data.has_valid('value'):
                node['axis'] = reduction_indices_data.value
            else:
                raise Error("Can not deduce `reduction_indices` for node {}. It should be deduced out of first port "
                            "input due to absence of value in this node".format(node.id))
            node.graph.remove_edge(reduction_indices_data.id, node.id)

        input_node = node.in_node()
        input_shape = np.array(node.in_node().shape, dtype=np.int64)
        output_node = node.out_node()

        # In case if axis is None it means that reduction comes along each dimension
        if node.axis is None:
            node.axis = int64_array(list(range(len(input_shape))))

        if not node.has_valid('reduce_type'):
            log.error('Reduce type for node {} not specified!'.format(node.id))
            return

        reduce_type = node.reduce_type
        if input_node.has_valid('value'):
            if reduce_type.lower() in ['mean', 'max']:
                # Value and Shape propagation for constant path
                output_node.value = Reduce.reduce_method_map[reduce_type.lower()](input_node.value,
                                                                                  axis=tuple(node.axis),
                                                                                  keepdims=node.keep_dims)
                output_node.shape = output_node.value.shape
            else:
                log.error('Reduce type {} is not supported for node {}'.format(reduce_type, node.id))
                return
        else:
            used_dims = np.zeros(len(input_shape), dtype=np.bool)
            output_shape = input_shape

            if node.axis.size == 1:
                node.axis = int64_array([node.axis.item()])

            for dim in node.axis:
                used_dims[dim] = True
                output_shape[dim] = 1

            # In case if keep dims == False, we should remove all 1 dims that was used in reduction
            if not node.keep_dims:
                output_shape = output_shape[np.invert(used_dims)]

            output_node.shape = output_shape
