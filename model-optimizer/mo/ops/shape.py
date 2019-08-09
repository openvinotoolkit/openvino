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

import numpy as np
import logging as log

from mo.graph.graph import Graph
from mo.ops.op import Op


class Shape(Op):
    op = 'ShapeOf'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node):
        if len(node.in_nodes()) != 1:
            log.warning('ShapeOf operation should have exact one input node, but it has {}'.format(len(node.in_nodes())))
            return

        if node.in_node(0).shape is not None:
            value = np.array(node.in_node(0).shape)
            node.out_node().shape = np.array(value.shape, dtype=np.int64)

            if not node.has_and_set('stop_value_propagation'):
                if node.has_valid('data_type'):
                    node.out_node().value = np.array(value, dtype=node.data_type)
                else:
                    node.out_node().value = np.array(value)
            node.out_node().shape = np.array(value.shape, dtype=np.int64)
        else:
            log.info('Can\'t infer shape and value for shape operation due to undefined input shape')

