# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class ShapeOfConstFolding(MiddleReplacementPattern):
    """
    The transformation calculates the value of the constant
    Const() -> ShapeOf() --> Const()
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('const', dict(op='Const')),
                ('const_data', dict(kind='data')),
                ('shape', dict(op='ShapeOf')),
            ],
            edges=[
                ('const', 'const_data', {}),
                ('const_data', 'shape', {})
            ])

    def replace_pattern(self, graph: Graph, match: [dict, SubgraphMatch]):
        shape = match['shape']
        if shape.has_valid('value'):
            shape_name = shape.soft_get('name', shape.id)
            shape_value = shape.value
            shape_const_node = Const(graph, {'name': shape_name + '/ExecutionConstValue',
                                             'value': shape_value}).create_node()
            shape.out_port(0).get_connection().set_source(shape_const_node.out_port(0))
            rename_nodes([(shape, shape_name + '/TBD'), (shape_const_node, shape_name)])
