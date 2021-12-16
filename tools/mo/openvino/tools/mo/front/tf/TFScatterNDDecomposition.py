# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.ops.scatternd import ScatterNDUpdate

class TFScatterNDDecomposition(FrontReplacementSubgraph):
    """
    Description
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for tf_scatter_nd in graph.get_op_nodes(op='TFScatterND'):
            if not tf_scatter_nd.is_in_port_connected(0) or not tf_scatter_nd.is_in_port_connected(1) \
                    or not tf_scatter_nd.is_in_port_connected(2):
                continue
            name = tf_scatter_nd.soft_get('name', tf_scatter_nd.soft_get('id'))
            indices_port = tf_scatter_nd.in_port(0).get_source()
            updates_port = tf_scatter_nd.in_port(1).get_source()
            shape_port = tf_scatter_nd.in_port(2).get_source()
            # need get type of  const type
            zero_const = Const(graph, {'value': float32_array(0.0), 'name': name + '/zero_const'}).create_node()
            broad_cast_node = Broadcast(graph, {'name': name + '/broadcast'}).create_node()
            broad_cast_node.in_port(0).connect(zero_const.out_port(0))
            broad_cast_node.in_port(1).connect(shape_port)
            scatter_nd_node = ScatterNDUpdate(graph, {'name': name + '/replaced'}).create_node()
            scatter_nd_node.in_port(0).connect(broad_cast_node.out_port(0))
            scatter_nd_node.in_port(1).connect(indices_port)
            scatter_nd_node.in_port(2).connect(updates_port)
            tf_scatter_nd.out_port(0).get_connection().set_source(scatter_nd_node.out_port(0))
            tf_scatter_nd.in_port(0).disconnect()
            tf_scatter_nd.in_port(1).disconnect()
            tf_scatter_nd.in_port(2).disconnect()
