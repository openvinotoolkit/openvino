# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class TensorArray(Op):
    op = "TensorArrayV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': TensorArray.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        size = node.in_node(0)
        assert size.value is not None

        # 0 port: handle
        if 0 in node.out_nodes().keys():
            if node.has_valid('element_shape'):
                element_shape = node['element_shape']
            else:
                element_shape = None

            out_node = node.out_node(0).id
            output_value = node.out_node(0).id
            node.graph.node[out_node]['value'] = mo_array(output_value)

            output_shape = node.graph.node[out_node]['value'].shape
            node.graph.node[out_node]['shape'] = shape_array(output_shape)

            node.graph.node[out_node]['element_shape'] = shape_array(element_shape)
            node.graph.node[out_node]['size'] = size.value
        # 1 port flow
        if 1 in node.out_nodes().keys():
            output_value = None

            out_node = node.out_node(1).id
            node.graph.node[out_node]['value'] = None if output_value is None else mo_array(output_value)
            node.graph.node[out_node]['shape'] = shape_array(output_shape)
