"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph


class AssignAndAssertElimination(FrontReplacementPattern):
    # The solution with removal of Assign and Assert operations is temporary.
    # The proper solution is to keep these operations until the partial inference
    # phase when control flow edges are properly handled and later unnecessary ones are eliminated.
    # In order to achieve this we need to implement control flow inference function
    # for these operations similar to "Merge" and "Switch" operations.
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes():
            if node.soft_get('op') in ["Assign", "AssignSub", "AssignAdd", "Assert"]:
                log.debug('"{}" op with id="{}" was removed'.format(node.op, node.id))
                graph.remove_node(node.id)
