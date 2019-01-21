"""
 Copyright (c) 2017-2018 Intel Corporation

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

from mo.graph.graph import Node
from mo.ops.op import Op, PermuteAttrs


class ReverseSequence(Op):
    op = 'ReverseSequence'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            #'type': not set, there shouldn't be translated to real layer
            'seq_dim': None,
            'batch_dim': None,
            'op': __class__.op,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
        ]

    @staticmethod
    def infer(node):
        if not node.has_valid('seq_dim'):
            assert 1 in node.in_nodes()
            assert node.in_node(1).has_valid('value')
            assert node.in_node(1).value.size == 1
            node['seq_dim'] = node.in_node(1).value.item()
            node.graph.remove_edge(node.in_node(1).id, node.id)
        assert len(node.out_nodes()) == 1
        node.out_node().shape = node.in_node().shape.copy()
