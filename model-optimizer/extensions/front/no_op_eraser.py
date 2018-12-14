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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import erase_node


class NoOpEraser(FrontReplacementSubgraph):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('noop', dict(kind='op', op='NoOp')),
                   ('output', dict(kind='op', op='OpOutput'))
                   ],
            edges=[('noop', 'output')]
        )

    @staticmethod
    def replace_sub_graph(graph: nx.MultiDiGraph, match: dict):
        erase_node(match['output'])
        erase_node(match['noop'])
        log.info("NoOp node \"{}\" was removed from the graph".format(match['noop'].id))
