# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class ExpandDimsToUnsqueeze(FrontReplacementPattern):
    """
    Converts the 'ExpandDims' layer to Unsqueeze layer with two inputs: the input with data and input with the
    dimensions to unsqueeze.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.Pack import Pack
        return [Pack]

    def find_and_replace_pattern(self, graph: Graph):
        for expand_dims_node in graph.get_op_nodes(op='ExpandDims'):
            if len(expand_dims_node.in_nodes()) == 1:
                expand_axis = expand_dims_node.expand_axis
                if not isinstance(expand_axis, np.ndarray):
                    expand_axis = int64_array([expand_axis]).flatten()
                unsqueeze_node = Unsqueeze(graph, {'name': expand_dims_node.id + '/Unsqueeze'}).create_node()
                unsqueeze_dims_node = Const(graph, {'name': expand_dims_node.id + '/Dims',
                                                    'value': expand_axis}).create_node()
                expand_dims_node.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
                expand_dims_node.out_port(0).get_connection().set_source(unsqueeze_node.out_port(0))
                unsqueeze_node.in_port(1).connect(unsqueeze_dims_node.out_port(0))
            elif len(expand_dims_node.in_nodes()) == 2:
                # For Unsqueeze-13 from ONNX
                expand_dims_name = expand_dims_node.soft_get('name', expand_dims_node.id)
                unsqueeze_node = Unsqueeze(graph, {'name': expand_dims_name  + '/Unsqueeze'}).create_node()
                rename_nodes([(expand_dims_node, expand_dims_name  + "/TBR"), (unsqueeze_node, expand_dims_name)])

                expand_dims_node.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
                expand_dims_node.in_port(1).get_connection().set_destination(unsqueeze_node.in_port(1))
                expand_dims_node.out_port(0).get_connection().set_source(unsqueeze_node.out_port(0))
            else:
                log.error('The ExpandDims node {} has wrong number of inputs'.format(expand_dims_node.soft_get('name')))
