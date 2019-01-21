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

from extensions.middle.ConstSwitchResolver import ConstSwitchEraser
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class UselessMergeEraser(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        return [ConstSwitchEraser]

    def pattern(self):
        return dict(
            nodes=[('merge', dict(kind='op', op='Merge'))],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        if len(graph.in_edges(match['merge'].id)) <= 1:
            remove_op_node_with_data_node(graph, match['merge'])
            log.info("Useles Merge op and data nodes was deleted op='{}'".format(match['merge'].id))
