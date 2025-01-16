# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined


class ShapeOfConstFolding(BackReplacementPattern):
    """
    The transformation folds ShapeOf(Const) -> Const
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.back.MatMulNormalizer import SmartReshape_HC_Reshape_MatMul
        return [SmartReshape_HC_Reshape_MatMul]

    def find_and_replace_pattern(self, graph: Graph):
        for shapeof_node in graph.get_op_nodes(op='ShapeOf'):
            in_node = shapeof_node.in_port(0).get_source().node
            if in_node.op == 'Const' or (shapeof_node.has_and_set('allow_fold') and is_fully_defined(shapeof_node.in_port(0).data.get_shape())):
                shapeof_node.in_port(0).disconnect()
                shape_name = shapeof_node.soft_get('name', shapeof_node.id)
                shape_value = shapeof_node.out_port(0).data.get_value()
                shape_const_node = Const(graph, {'name': shape_name + '/ExecutionConstValue',
                                                 'value': shape_value}).create_node()
                shapeof_node.out_port(0).get_connection().set_source(shape_const_node.out_port(0))
                rename_nodes([(shapeof_node, shape_name + '/TBD'), (shape_const_node, shape_name)])
