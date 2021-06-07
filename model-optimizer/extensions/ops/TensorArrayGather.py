# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import is_fully_defined, dynamic_dimension, shape_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.utils import symm_match_shapes


class TensorArrayGather(Op):
    op = "TensorArrayGatherV3"

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': TensorArrayGather.array_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def array_infer(node: Node):
        assert len(node.in_nodes()) == 3

        handle = node.in_node(0)

        ta_node = Node(node.graph, str(handle.value))

        if ta_node.has_valid('element_shape') and ta_node.element_shape is not None and len(ta_node.element_shape) > 0:
            assert symm_match_shapes(ta_node['element_shape'], node.element_shape)
        else:
            ta_node['element_shape'] = node.element_shape
        data_shape = ta_node['element_shape']
        # assert is_fully_defined(data_shape) or (data_shape.size == 2 and data_shape[0] is dynamic_dimension and
        #                                         data_shape[1] is not dynamic_dimension)
        assert ta_node.has_valid('size')
        size = ta_node['size']

        output_shape = [size] + [data_shape[i] for i in range(len(data_shape))]

        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = shape_array(output_shape)
