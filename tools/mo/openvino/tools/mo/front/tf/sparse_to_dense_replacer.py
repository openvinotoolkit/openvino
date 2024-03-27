# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Node, Graph, rename_nodes
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.scatternd import ScatterNDUpdate


class SparseToDenseReplacer(FrontReplacementOp):
    """
    This replacer substitutes TensorFlow SparseToDense operation with Broadcast -> ScatterND chain.
    The Broadcast operation creates a tensor filled with default value and of required shape.
    The ScatterND operation updates the created tensor with required values at required locations.
    """
    op = "SparseToDense"
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
        from openvino.tools.mo.front.tf.CTCLossReplacement import CTCLossReplacement
        return [CTCGreedyDecoderReplacement, CTCLossReplacement]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)

        # broadcast default value to required shape
        broadcast_node = Broadcast(graph, {'name': node_name + '/Broadcast_'}).create_node()
        node.in_port(1).get_connection().set_destination(broadcast_node.in_port(1))
        if not node.in_port(3).disconnected():
            node.in_port(3).get_connection().set_destination(broadcast_node.in_port(0))
        else:
            broadcast_node.in_port(0).connect(Const(graph, {'name': broadcast_node.name + '/FillValue_',
                                                            'value': np.float32(0)}
                                                    ).create_node().out_port(0))

        # update broadcasted tensor with required values at required locations
        scatternd_node = ScatterNDUpdate(graph, {'name': node_name + '/ScatterNDUpdate_'}).create_node()
        scatternd_node.in_port(0).connect(broadcast_node.out_port(0))
        node.in_port(0).get_connection().set_destination(scatternd_node.in_port(1))
        node.in_port(2).get_connection().set_destination(scatternd_node.in_port(2))

        rename_nodes([(node, node_name + "/AbandonedName"), (scatternd_node, node_name)])

        return [scatternd_node.id]
