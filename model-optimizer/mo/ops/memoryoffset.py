"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph, Node
from mo.ops.op import Op
from mo.utils.error import Error


class MemoryOffset(Op):
    op = 'MemoryOffset'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': 'MemoryOffset',
            'pair_name': None,
            'has_default': False,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['t']

    @staticmethod
    def infer(node: Node):
        # MemoryOffset is splitted in 2 parts to avoid cycle in graph
        # Calculate shape from shape of previous layer where possible
        # In other cases information about shapes from initial Kaldi model used
        if len(node.in_nodes()) > 0:
            copy_shape_infer(node)
            pair_node = Node(node.graph, node.pair_name)
            for out_node_name, params in pair_node.get_outputs():
                out_node = Node(node.graph, out_node_name)
                out_node.shape = node.out_node().shape
        else:
            pair_node = Node(node.graph, node.pair_name)
            if pair_node.in_node().shape is not None:
                for out_node_name, params in node.get_outputs():
                    out_node = Node(node.graph, out_node_name)
                    out_node.shape = pair_node.in_node().shape
                copy_shape_infer(pair_node)
            elif pair_node.has_valid('element_size'):
                # TODO Add here real batch
                for out_node_name, params in node.get_outputs():
                    out_node = Node(node.graph, out_node_name)
                    out_node.shape = np.array([1, pair_node['element_size']])
            elif pair_node.in_node().in_node().op == 'FullyConnected':
                out_size = pair_node.in_node().in_node()['out-size']
                for out_node_name, params in node.get_outputs():
                    out_node = Node(node.graph, out_node_name)
                    out_node.shape = np.array([1, out_size])
            elif pair_node.in_node().in_node().op == 'Normalize':
                    out_size = pair_node.in_node().in_node()['in_dim']
                    for out_node_name, params in node.get_outputs():
                        out_node = Node(node.graph, out_node_name)
                        out_node.shape = np.array([1, out_size])
            else:
                raise Error("Can't calculate MemoryOffset shape for node {}. Possibly you need to add shape for it through --input_shape".format(node.id))
