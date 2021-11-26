# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.slice import Slice


class AttributedSliceToSliceReplacer(FrontReplacementOp):
    """
    This class replaces AttributedSlice -> Slice
    """
    op = 'AttributedSlice'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        slice_name = node.soft_get('name', node.id)

        slice_node = create_op_with_const_inputs(graph, Slice, {1: node.starts, 2: node.ends, 3: node.axes})
        rename_nodes([(node, slice_name + '/to_be_removed'), (slice_node, slice_name)])

        node.in_port(0).get_connection().set_destination(slice_node.in_port(0))
        node.out_port(0).get_connection().set_source(slice_node.out_port(0))
