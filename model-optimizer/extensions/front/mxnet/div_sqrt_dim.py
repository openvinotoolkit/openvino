# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.elementwise import Div
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.power import AttributedPower
from mo.ops.shape import Shape
from mo.utils.shape import get_shape_values_by_range_idxs


class DivSqrtDim(FrontReplacementOp):
    """
    Convert div_sqrt_dim op to div / sqrt(shapeof(-1))
    """
    op = '_contrib_div_sqrt_dim'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        div_sqrt = match['op']
        shape_node = Shape(graph, dict(name=div_sqrt.id + '/Shape')).create_node()
        rank_node = Shape(graph, dict(name=div_sqrt.id + '/RankShape')).create_node()
        shape_node.in_port(0).connect(div_sqrt.in_port(0).get_source())
        rank_node.in_port(0).connect(shape_node.out_port(0))

        shape_values_node = get_shape_values_by_range_idxs(shape=shape_node, rank=rank_node, begin=-2,
                                                           end=-1,
                                                           include_begin=False, include_end=True)

        pow_node = AttributedPower(graph, dict(name=div_sqrt.id + '/Sqrt', power=0.5)).create_node()
        div_node = Div(graph, dict(name=div_sqrt.id + "/DivSqrt")).create_node()

        pow_node.in_port(0).connect(shape_values_node.out_port(0))
        div_sqrt.in_port(0).get_connection().set_destination(div_node.in_port(0))
        div_node.in_port(1).connect(pow_node.out_port(0))
        div_sqrt.out_port(0).get_connection().set_source(div_node.out_port(0))
