# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from extensions.middle.pass_separator import PostMiddleStart, MiddleFinish
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.passes.fusing.helpers import get_next_operation
from mo.middle.replacement import MiddleReplacementPattern


class FuseReshapesSequence(MiddleReplacementPattern):
    """
    Finds sequence of Reshapes operations and merge them to a single Reshape operation.
    """
    # TODO the pass should be extended for Reshape with special symbols "0" or "-1"
    # For example: 1,100 -> Reshape(2,5,10) -> 2,5,10 -> Reshape(0,10,-1) -> 2,10,5

    enabled = True
    run_not_recursively = True  # non-unified data nodes view in TI body (no Const ops, bare data node)

    def run_before(self):
        return [PostMiddleStart]

    def run_after(self):
        return [MiddleFinish]

    def find_and_replace_pattern(self, graph: Graph):
        reshape_nodes = graph.get_op_nodes(type='Reshape')
        for node in reshape_nodes:
            if not graph.has_node(node.id):
                # the Reshape node has been removed in the previous iteration
                continue
            if len(node.out_port(0).get_destinations()) == 1:
                log.debug('First phase for Reshape: {}'.format(node.soft_get('name')))

                next_op = get_next_operation(node)[0]
                log.debug('second node: id={}, type={}'.format(next_op.soft_get('id'), next_op.soft_get('type')))
                if next_op.has_valid('type') and next_op.type == 'Reshape':
                    dim_value = next_op.in_port(1).data.get_value()
                    if dim_value is None or 0 in dim_value or -1 in dim_value:
                        # we do not fuse reshape sequences with special symbols: 0, -1
                        continue

                    # Detected Reshape1 --> data --> Reshape2 pattern without side edges. Remove Reshape1
                    log.debug('Second phase for Reshape: {}'.format(node.soft_get('name')))
                    remove_op_node_with_data_node(graph, node)
