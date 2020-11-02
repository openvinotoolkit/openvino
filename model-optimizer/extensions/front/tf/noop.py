"""
 Copyright (C) 2020 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.utils.error import Error


class NoOpElimination(FrontReplacementOp):
    """
    NoOp does nothing and it has no data flow edges.
    It operates only with control flow edges.
    """
    op = "NoOp"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        in_edges = node.in_edges()
        out_edges = node.out_edges()
        if len(in_edges) == 0 and len(out_edges) == 0:
            graph.remove_node(node.id)
            log.debug('NoOp op was removed {}'.format(node.id))
        else:
            raise Error('NoOp node {} contains data flow edges'.format(node.id))
