# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.scatternd import ScatterNDUpdate
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph, rename_nodes
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const


class SparseToDenseReplacer(FrontReplacementOp):
    """
    This replacer substitutes TensorFlow SparseToDense operation with Broadcast -> ScatterND chain.
    The Broadcast operation creates a tensor filled with default value and of required shape.
    The ScatterND operation updates the created tensor with required values at required locations.
    """
    op = "SparseToDense"
    enabled = True

    def run_after(self):
        from extensions.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement
        from extensions.front.tf.CTCLossReplacement import CTCLossReplacement
        return [CTCGreedyDecoderReplacement, CTCLossReplacement]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)

        # broadcast default value to required shape
        broadcast_node = Broadcast(graph, {'name': node_name + '/Broadcast_'}).create_node()
        node.in_port(1).get_connection().set_destination(broadcast_node.in_port(1))
        if not node.in_port(3).disconnected():
            # TODO: remove casting once we start to support I64 model input
            # cast default value to I32 due limitation about I64 input support
            # so that input parameter and default value will be of the same I32 type as required ScatterNDUpdate
            cast_default_value = Cast(graph, {'name': node_name + '/CastDefaultValue', 'dst_type': np.int32}).create_node()
            node.in_port(3).get_connection().set_destination(cast_default_value.in_port(0))
            broadcast_node.in_port(0).connect(cast_default_value.out_port(0))
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
