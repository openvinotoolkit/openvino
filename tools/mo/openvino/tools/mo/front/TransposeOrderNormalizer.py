# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.tf.pad_tf_to_pad import PadTFToPad
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error


class TransposeOrderNormalizer(FrontReplacementSubgraph):
    """
    Transpose operation requires information about order, that is represented in original frameworks differently:
        - by layer parameter
        - by 1-port input value

    TransposeOrderNormalizer reforms Transpose operations to store axis info in 1-port input.
    """
    enabled = True

    def run_before(self):
        # refer to the comments of the ObjectDetectionAPIPreprocessorReplacement transformation in the
        # <MO_DIR>/openvino/tools/mo/front/tf/ObjectDetectionAPI.py file for more details why this dependency is needed.
        return [PadTFToPad]

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(type='Transpose'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['transpose']
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) == 1:
            if node.has_valid('order'):
                const = Const(graph, {'value': node.order}).create_node()
                node.add_input_port(1, skip_if_exist=True)
                const.out_port(0).connect(node.in_port(1))
                del graph.node[node.id]['order']
            elif node.has('order') and node.order is None:
                assert node.has_and_set('reverse_order')
            else:
                raise Error('Can not deduce transpose `order` for {}: only one in_port and no `order` parameter.'
                            ''.format(node.soft_get('name', node.id)))
