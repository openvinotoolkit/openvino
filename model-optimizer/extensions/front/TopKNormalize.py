# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class TopKNormalize(FrontReplacementPattern):
    """
    This pass do TopK layer normalization:
        1. Adds the second input to the TopK layer if it has just one. In this case the attribute 'k' should be defined.
        2. If one of TopK ports isn't connected - adds output on this port to keep this port in IR.

    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for topk_node in graph.get_op_nodes(op='TopK'):
            if topk_node.in_port(1).disconnected():
                assert topk_node.has_valid('k'), 'The TopK node "{}" misses "k" attribute'.format(topk_node.name)
                k_node = Const(graph, {'name': topk_node.id + '/Dims', 'value': int64_array(topk_node.k)}).create_node()
                topk_node.in_port(1).connect(k_node.out_port(0))
                del topk_node['k']
            else:
                log.debug('The TopK node input "{}" is already normalized'.format(topk_node.name))
