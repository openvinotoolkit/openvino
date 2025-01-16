# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error


class SqueezeNormalize(FrontReplacementPattern):
    """
    Normalizes inputs of the Squeeze layers. The layers should have two inputs: the input with data and input with the
    dimensions to squeeze. If the second input is omitted then all dimensions of size 1 should be removed.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for squeeze_node in graph.get_op_nodes(op='Squeeze'):
            if len(squeeze_node.in_nodes()) == 1 and squeeze_node.has_valid('squeeze_dims'):
                dims_node = Const(graph, {'name': squeeze_node.id + '/Dims',
                                          'value': int64_array(squeeze_node.squeeze_dims)}).create_node()
                squeeze_node.in_port(1).connect(dims_node.out_port(0))
                del squeeze_node['squeeze_dims']
            elif len(squeeze_node.in_nodes()) == 2:
                log.debug('The Squeeze node "{}" is already normalized'.format(squeeze_node.name))
            else:
                raise Error('The Squeeze layer "{}" should either have 2 inputs or one input and an "squeeze_dims" '
                            'attribute'.format(squeeze_node.soft_get('name')))
