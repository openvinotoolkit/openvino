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

import logging as log

import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.utils.error import Error


class AssignElimination(FrontReplacementOp):
    op = "Assign"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        # here we request all data flow output edges (control flow edges will not be listed)
        out_edges = node.out_edges()
        if len(out_edges) == 0:
            graph.remove_node(node.id)
            log.debug('Assign op was removed {}'.format(node.id))
        else:
            raise Error('Data flow edge coming out of Assign node {}'.format(node.id))


class AssignSubElimination(FrontReplacementOp):
    op = "AssignSub"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        # here we request all data flow output edges (control flow edges will not be listed)
        out_edges = node.out_edges()
        if len(out_edges) == 0:
            graph.remove_node(node.id)
            log.debug('AssignSub op was removed {}'.format(node.id))
        else:
            raise Error('Data flow edge coming out of AssignSub node {}'.format(node.id))


class AssignAddElimination(FrontReplacementOp):
    op = "AssignAdd"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        # here we request all data flow output edges (control flow edges will not be listed)
        out_edges = node.out_edges()
        if len(out_edges) == 0:
            graph.remove_node(node.id)
            log.debug('AssignAdd op was removed {}'.format(node.id))
        else:
            raise Error('Data flow edge coming out of AssignAdd node {}'.format(node.id))


class AssertElimination(FrontReplacementOp):
    op = "Assert"
    enabled = True

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        node = match['op']
        # here we request all data flow output edges (control flow edges will not be listed)
        out_edges = node.out_edges()
        if len(out_edges) == 0:
            graph.remove_node(node.id)
            log.debug('Assert op was removed {}'.format(node.id))
        else:
            raise Error('Data flow edge coming out of Assert node {}'.format(node.id))
