# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_node, Node
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.pad import Pad


class ONNXPadToPad(FrontReplacementOp):
    """
    This transformation converts ONNXPad operation (ONNX semantic) to Pad operation (OpenVINO semantic).
    Refer to the Op implementation for the operations semantics description.
    """
    op = 'ONNXPad'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        # save the original node name to use it in the new Pad op instance
        original_name = node.soft_get('name', node.id)
        rename_node(node, original_name + '/TBR')

        new_pad = Pad(graph, {'mode': node.soft_get('mode', None)}).create_node()
        rename_node(new_pad, original_name)

        node.in_port(0).get_connection().set_destination(new_pad.in_port(0))

        if node.soft_get('mode') == 'constant':
            # the input with fill value is an optional third input in ONNX
            if not node.in_port(2).disconnected():
                node.in_port(2).get_connection().set_destination(new_pad.in_port(3))
            else:
                new_pad.in_port(3).connect(Const(graph, {'value': 0.0}).create_node().out_port(0))

        # convert ONNX representation of the pads as [2 * N] to MO representation: [N] and [N]
        split_pads = create_op_with_const_inputs(graph, Split, {1: int64_array(0)}, {'num_splits': 2})
        node.in_port(1).get_connection().set_destination(split_pads.in_port(0))
        split_pads.out_port(0).connect(new_pad.in_port(1))
        split_pads.out_port(1).connect(new_pad.in_port(2))

        return [new_pad.id]
