# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.scatter import ScatterElementsUpdate
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape


class MaxUnpoolFrontReplacer(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ("max_pool0", dict(op="MaxPool")),
                ("max_pool1", dict(op="MaxPool")),
                ("slice", dict(op="AttributedSlice")),
                ("sub", dict(op="Sub")),
                ("unpool", dict(op="max_unpool2d")),
            ],
            edges=[
                ("max_pool1", "slice"),
                ("max_pool0", "sub", {"in": 0}),
                ("slice", "sub", {"in": 1}),
                ("sub", "unpool", {"in": 1}),
            ],
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        max_pool = match["max_pool0"]
        node_name = max_pool.soft_get("name", max_pool.id)

        unpool = match["unpool"]
        unpool_inp = unpool.in_port(0).get_source().node

        max_pool_input = max_pool.in_port(0).get_source().node

        zero_const = Const(graph, {"value": 0.0}).create_node()
        zero_inp = Mul(graph, {}).create_node([max_pool_input, zero_const])

        shape_1d = Const(graph, {"value": np.array([-1], dtype=np.int64)}).create_node()

        index_new_shape = Reshape(graph, {"special_zero": True}).create_node()
        index_new_shape.in_port(0).get_connection().set_source(max_pool.out_port(1))
        index_new_shape.in_port(1).get_connection().set_source(shape_1d.out_port(0))

        out_new_shape = Reshape(graph, {"special_zero": True}).create_node(
            [unpool_inp, shape_1d]
        )
        zero_inp_new_shape = Reshape(graph, {"special_zero": True}).create_node(
            [zero_inp, shape_1d]
        )

        shape_node = Const(graph, {"value": 0}).create_node()
        scatter_node = ScatterElementsUpdate(
            graph, {"name": node_name + "/ScatterElementsUpdate_"}
        ).create_node([zero_inp_new_shape, index_new_shape, out_new_shape, shape_node])

        origin_shape = Shape(graph, {"name": "Shape"}).create_node([max_pool_input])
        out_origin_shape = Reshape(graph, {"special_zero": True}).create_node(
            [scatter_node, origin_shape]
        )

        unpool.out_port(0).get_connection().set_source(out_origin_shape.out_port(0))
