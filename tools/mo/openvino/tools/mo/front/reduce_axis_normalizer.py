# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import reduce_map
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class ReduceAxisNormalizer(FrontReplacementSubgraph):
    """
    Reduce operation requires information about axis, that is represented in original frameworks differently: as an
    operation attribute or as a 1-st input port value. ReduceAxisNormalizer adds second input to Reduce operations with
    axes to normalize if axes are specified as an attribute.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', op=lambda op: op in reduce_map))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['reduce']
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) == 1:
            node_name = node.soft_get('name', node.id)

            # if the 'axis' is None then we still add a second input to the layer with a 1D array with 1 element equal
            # to None. The infer function handles this case because the input shape is known at this stage only
            if node.has_valid('axis'):
                const = Const(graph, {'name': node_name + '/axis', 'value': node.axis}).create_node()
                node.add_input_port(1, skip_if_exist=True)
                const.out_port(0).connect(node.in_port(1))
                del graph.node[node.id]['axis']
            else:
                # The default (if there is no 'axis') is to reduce over all the dimensions of the input tensor.
                axes = create_op_with_const_inputs(graph, Range, {0: int64_array(0), 2: int64_array(1)},
                                                   dict(name=node_name + '/axes'))
                end_of_range = Rank(graph, dict(name=node_name + '/range_end')).create_node()
                node.in_port(0).get_connection().get_source().connect(end_of_range.in_port(0))
                end_of_range.out_port(0).connect(axes.in_port(1))

                node.add_input_port(1, skip_if_exist=True)
                axes.out_port(0).connect(node.in_port(1))
