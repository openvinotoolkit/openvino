# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceProd
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.shape import Shape


class SizeFrontReplacer(FrontReplacementOp):
    """
    Replace Size op by Shape -> ReduceProd operations
    """
    op = "Size"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        name = node.soft_get('name', node.id)
        assert node.has_valid('output_type'), \
            'Size node should have `output_type` attribute, but it`s not for node {}'.format(name)

        shape = Shape(graph, {'name': name + '/Shape/', 'output_type': node.output_type}).create_node()
        node.in_port(0).get_connection().set_destination(shape.in_port(0))
        reduce_prod = create_op_node_with_second_input(
            graph, ReduceProd, int64_array([0]), {'name': shape.name + 'ReduceProd/', 'keep_dims': False}, shape)
        node.out_port(0).get_connection().set_source(reduce_prod.out_port(0))

        rename_nodes([(node, name + '/ToBeDeleted'), (reduce_prod, name)])
