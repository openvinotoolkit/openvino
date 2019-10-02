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

import networkx as nx

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class SquaredDifference(Op):
    op = 'SquaredDifference'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,  # IE layer type, not required if this op won't be dumped to IE
            'op': __class__.op,  # internal MO name for the operation, can be the same as type; required
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': lambda node: eltwise_infer(node, lambda a, b: (a - b) ** 2)},
                         attrs)
