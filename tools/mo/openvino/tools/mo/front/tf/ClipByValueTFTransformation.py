# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.ops.elementwise import Minimum, Maximum
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class ClipByValueTFTransformation(FrontReplacementSubgraph):
    """
    The transformation replaces the ClipByValueTF operation which works as Clamp but supports broadcasting of inputs
    with Minimum and Maximum.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for cbv in graph.get_op_nodes(op='ClipByValueTF'):
            cbv_name = cbv.soft_get('name', cbv.id)
            minimum = Minimum(graph, {'name': cbv_name + '/CLipMinimum'}).create_node()
            maximum = Maximum(graph, {'name': cbv_name + '/CLipMaximum'}).create_node()
            minimum.in_port(0).connect(cbv.in_port(0).get_source())
            minimum.in_port(1).connect(cbv.in_port(2).get_source())
            maximum.in_port(0).connect(minimum.out_port(0))
            maximum.in_port(1).connect(cbv.in_port(1).get_source())
            cbv.out_port(0).get_connection().set_source(maximum.out_port(0))

            rename_nodes([(cbv, cbv_name + '/TBR'), (maximum, cbv_name)])
            graph.remove_node(cbv.id)
