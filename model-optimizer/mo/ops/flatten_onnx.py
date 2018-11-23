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

import networkx as nx
import numpy as np
import logging as log

from mo.ops.op import Op


class FlattenONNX(Op):
    op = 'Flatten'
    enabled = True

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'type': 'Reshape',
            'op': __class__.op,
            'infer': __class__.infer,
        }, attrs)

    def supported_attrs(self):
        return ['axis', ('dim', lambda node: ','.join(map(str, node['dim'])))]

    @staticmethod
    def infer(node):
        """
        Infers shape of flatten node as it is done in onnx.
        Args:
            node: graph flatten node
        """
        if not node.has_valid('axis'):
            log.debug('Can\'t calculate output shape for {} node due to missing axis attribute'.format(node.name))
            return

        if node.in_node(0).shape is None:
            log.debug('Can\'t calculate output shape for {} node due to shape for input node is None'.format(node.name))
            return

        if len(node.in_nodes()) != 1:
            log.debug('Can\'t calculate output shape for {} node. Number of input nodes should be equal 1 instead of {}'.format(node.name, len(node.in_nodes())))
            return

        axis = node.axis
        shape = node.in_node(0).shape
        dim = [np.prod(shape[0:axis]), np.prod(shape[axis:])]
        node['dim'] = np.array(dim)
        node.out_node().shape = np.array(dim)
        if node.in_node(0).has_valid('value'):
            node.out_node().value = node.in_node(0).value
            node.out_node().value.shape = np.array(dim)
