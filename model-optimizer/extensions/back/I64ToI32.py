"""
 Copyright (C) 2018-2020 Intel Corporation

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
import numpy as np

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from mo.back.replacement import BackReplacementPattern
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph


class I64ToI32(BackReplacementPattern):
    """
    Change the Cast to int64 to int32 since not all plugins support int64 data type for non IR V10 case.
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        return [ForceStrictPrecision]

    def pattern(self):
        return dict(
            nodes=[
                ('cast', dict(op='Cast'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['cast']
        if node.dst_type == np.int64:
            log.warning('Change data type from {} to {} for node {}'.format(node.dst_type, np.int32, node.name))
            node.dst_type = np.int32
