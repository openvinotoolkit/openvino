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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class NormalizeL2Normalize(FrontReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for normalize_l2 in graph.get_op_nodes(op='NormalizeL2'):
            if normalize_l2.in_port(1).disconnected():
                assert normalize_l2.has_valid('axis'), 'The NormalizeL2 node "{}" misses "axis" attribute.' \
                                                       ''.format(normalize_l2.name)
                axis_node = Const(graph, {'name': normalize_l2.id + '/Axis',
                                          'value': int64_array([normalize_l2.axis])}).create_node()
                normalize_l2.in_port(1).connect(axis_node.out_port(0))
                del normalize_l2['axis']
            else:
                log.debug('The NormalizeL2 node input "{}" is already normalized'.format(normalize_l2.name))
