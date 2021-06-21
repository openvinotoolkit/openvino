# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.ops.elementwise import Add, Pow
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.passes.convert_data_type import data_type_str_to_np


class ComplexAbsAfterComplex(FrontReplacementSubgraph):
    """
    This transformation converts a sub-graph

    SomeOp1    SomeOp2
       |          |
       ------------
            |
         Complex
            |
        ComplexAbs

    into the sub-graph

                SomeOp1      SomeOp2
                   |           |
     Constant[2]--Pow         Pow--Constant[2]
                   |           |
                   -------------
                        Add
                         |
                        Pow--Constant[0.5]
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('complex', dict(op='Complex')),
                ('abs', dict(op='ComplexAbs')),
            ],
            edges=[
                ('complex', 'abs', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        cmp = match['complex']
        complex_abs = match['abs']
        complex_abs_name = complex_abs.soft_get('name', complex_abs.id)

        power_type = data_type_str_to_np(graph.graph['cmd_params'].data_type)

        pow0 = create_op_with_const_inputs(graph, Pow, {1: power_type(2.0)},
                                           {'name': complex_abs_name + '/real_part_squared'})
        pow1 = create_op_with_const_inputs(graph, Pow, {1: power_type(2.0)},
                                           {'name': complex_abs_name + '/imag_part_squared'})

        cmp.in_port(0).get_connection().set_destination(pow0.in_port(0))
        cmp.in_port(1).get_connection().set_destination(pow1.in_port(0))

        add = Add(graph, {'name': complex_abs_name + '/squared_abs'}).create_node([pow0, pow1])
        sqrt = create_op_with_const_inputs(graph, Pow, {1: power_type(0.5)}, {})
        add.out_port(0).connect(sqrt.in_port(0))

        complex_abs.out_port(0).get_connection().set_source(sqrt.out_port(0))

        rename_nodes([(complex_abs, complex_abs_name + '/to_be_removed'), (sqrt, complex_abs_name)])
