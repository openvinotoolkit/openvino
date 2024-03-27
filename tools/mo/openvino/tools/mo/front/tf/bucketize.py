# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class BucketizeFrontReplacer(FrontReplacementSubgraph):
    """
    Moves the boundaries data from attribute to the second input tensor.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for bucketize in graph.get_op_nodes(op='Bucketize'):
            if bucketize.in_port(1).disconnected():
                assert bucketize.has_valid('boundaries'), 'The Bucketize node "{}" misses "boundaries" attribute'.format(bucketize.name)
                boundaries_node = Const(graph, {'name': bucketize.name + '/Bucketize_boundaries_', 'value': bucketize.boundaries}).create_node()
                bucketize.in_port(1).connect(boundaries_node.out_port(0))
                del bucketize['boundaries']
            else:
                log.debug('The Bucketize node input "{}" is already normalized'.format(bucketize.name))
