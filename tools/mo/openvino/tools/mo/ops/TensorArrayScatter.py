# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.utils import match_shapes


class TensorArrayScatter(Op):
    op = "TensorArrayScatterV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': TensorArrayScatter.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        handle = node.in_node(0)
        value = node.in_node(2)
        flow_in = node.in_node(3)

        ta_node = Node(node.graph, str(handle.value))
        if ta_node.has_valid('element_shape') and len(ta_node.element_shape) > 0:
            assert match_shapes(ta_node['element_shape'], value.shape[1:]), \
                'Shapes are not compatible: {} and {}'.format(ta_node['element_shape'], value.shape[1:])
        else:
            ta_node['element_shape'] = value.shape[1:]

        # Assign element_shape anyway, because the original element_shape can contain -1
        ta_node['element_shape'] = value.shape[1:]

        output_value = flow_in.value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = shape_array(flow_in.shape)
            node.graph.node[out_node]['value'] = None if output_value is None else mo_array(output_value)
