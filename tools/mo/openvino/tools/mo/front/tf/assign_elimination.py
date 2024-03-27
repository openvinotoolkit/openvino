# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph


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
