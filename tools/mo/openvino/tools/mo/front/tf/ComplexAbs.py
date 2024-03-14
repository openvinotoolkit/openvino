# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.ops.elementwise import Pow
from openvino.tools.mo.ops.ReduceOps import ReduceSum
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np


class ComplexAbs(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.tf.ComplexAbsAfterComplex import ComplexAbsAfterComplex
        return [ComplexAbsAfterComplex]

    def find_and_replace_pattern(self, graph: Graph):
        for complex_abs in graph.get_op_nodes(op='ComplexAbs'):
            complex_abs_name = complex_abs.soft_get('name', complex_abs.id)
            power_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

            squared = create_op_with_const_inputs(graph, Pow, {1: power_type(2.0)},
                                                  {'name': complex_abs_name + '/squared_parts'})
            complex_abs.in_port(0).get_connection().set_destination(squared.in_port(0))
            sum = create_op_with_const_inputs(graph, ReduceSum, {1: int64_array(-1)},
                                              {'name': complex_abs_name + '/squared_abs'},
                                              squared)
            sqrt = create_op_with_const_inputs(graph, Pow, {1: power_type(0.5)}, {}, sum)

            complex_abs.out_port(0).get_connection().set_source(sqrt.out_port(0))

            rename_nodes([(complex_abs, complex_abs_name + '/to_be_removed'), (sqrt, complex_abs_name)])
