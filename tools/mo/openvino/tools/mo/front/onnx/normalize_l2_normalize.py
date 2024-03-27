# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


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
