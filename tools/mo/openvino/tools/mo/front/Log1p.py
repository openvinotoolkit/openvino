# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.activation_ops import Log
from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.ops.const import Const


class Log1p(FrontReplacementOp):
    """
    Log1p computes natural logarithm of (1 + x) element-wise.
    It replaces Log1p operation with Add -> Log.
    """
    op = "Log1p"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        const_dtype = np.float32
        if node.has_valid('data_type'):
            const_dtype = node.data_type
        const = Const(graph, {'value': mo_array([1], dtype=const_dtype)}).create_node()
        add = Add(graph, {'name': node.name + '/Add_'}).create_node()
        log = Log(graph, {'name': node.name + '/Log_'}).create_node()

        # Connect nodes: input -> Add -> Log
        const.out_port(0).connect(add.in_port(0))
        node.in_port(0).get_connection().set_destination(add.in_port(1))
        add.out_port(0).connect(log.in_port(0))
        rename_nodes([(node, node_name + '/delete'), (log, node_name)])

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [log.id]
