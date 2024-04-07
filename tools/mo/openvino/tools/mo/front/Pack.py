# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph, rename_nodes
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class Pack(FrontReplacementOp):
    op = "Pack"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        out_node = Concat(graph, {'axis': node.axis, 'in_ports_count': len(node.in_ports())}).create_node()
        pack_name = node.soft_get('name', node.id)

        for ind in node.in_ports():
            unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([node.axis])},
                                                         {'name': node.soft_get('name', node.id) + '/Unsqueeze'})
            node.in_port(ind).get_connection().set_destination(unsqueeze_node.in_port(0))
            unsqueeze_node.out_port(0).connect(out_node.in_port(ind))

        rename_nodes([(node, pack_name + '/TBR'), (out_node, pack_name)])
        return [out_node.id]

