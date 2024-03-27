# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_node


class Sub(FrontReplacementPattern):
    # This transformation is called directly from the 'openvino/tools/mo/middle/fusings.py' transformation
    enabled = False

    @staticmethod
    def sub_to_add_replacement(sub: Node):
        # we execute this transformation for V10 IR later on middle phase despite graph_condition
        # so we prevent Sub replacement on shape-calculating sub-graphs
        if sub.in_port(0).data.get_value() is not None and sub.in_port(1).data.get_value() is not None:
            return

        graph = sub.graph
        name = sub.soft_get('name', sub.id)

        # keep Add name the same as Sub -- because of mathematical equality of output tensors
        rename_node(node=sub, name=name + '/to_be_removed')

        # reconnect Sub in(out)puts to Add
        add = Add(graph, {'name': name}).create_node()
        rename_node(add, name)

        sub.in_port(0).get_connection().set_destination(add.in_port(0))
        sub.in_port(1).get_connection().set_destination(add.in_port(1))
        sub.out_port(0).get_connection().set_source(add.out_port(0))

        # restore mathematical equivalence to Sub operation: Sub(A, B) = Add(A, Mul(B, -1))
        const_dtype = sub.soft_get('data_type', np.float32)
        negate = create_op_with_const_inputs(graph, Mul, {1: mo_array(-1, dtype=const_dtype)}, {'name': name + '/neg_'})
        add.in_port(1).get_connection().insert_node(negate)

    def find_and_replace_pattern(self, graph: Graph):
        for sub in graph.get_op_nodes(op='Sub'):

            # The attribute zero_point_sub indicates that the node can be used in ConvertQuantizeDequantize
            # transformation (offline transformations). Pattern of such transformation expects Subtract node.
            if sub.has_and_set('zero_point_sub'):
                continue
            self.sub_to_add_replacement(sub)
