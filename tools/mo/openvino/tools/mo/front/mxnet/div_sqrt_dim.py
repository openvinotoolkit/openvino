# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.front.PowerToEltwises import PowerToEltwises
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.ConvertLike import ConvertLike
from openvino.tools.mo.ops.elementwise import Div
from openvino.tools.mo.ops.power import AttributedPower
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices


class DivSqrtDim(FrontReplacementOp):
    """
    Replace _contrib_div_sqrt_dim with sub-graph that matches the formula out = (data / sqrt(data.shape[-1]))
    """
    op = '_contrib_div_sqrt_dim'
    enabled = True

    def run_before(self):
        return [PowerToEltwises]

    def replace_sub_graph(self, graph: Graph, match: dict):
        div_sqrt = match['op']
        div_sqrt_name = div_sqrt.soft_get('name', div_sqrt.id)
        shape_node = Shape(graph, dict(name=div_sqrt_name + '/Shape')).create_node()
        data_out_port = div_sqrt.in_port(0).get_source()
        shape_node.in_port(0).connect(data_out_port)

        shape_values_node = node_to_get_shape_value_of_indices(shape_node=shape_node, indices=[-1])

        pow_node = AttributedPower(graph, dict(name=div_sqrt_name + '/Sqrt',
                                               power=mo_array(0.5))).create_node()

        # Due to specification, Power must have inputs with the same data type.
        convert_pow_input = Cast(graph, dict(dst_type=np.float32,
                                             name=shape_values_node.name + '/ConvertToFP32')).create_node()
        div_node = Div(graph, dict(name="Div")).create_node()

        shape_values_node.out_port(0).connect(convert_pow_input.in_port(0))
        convert_pow_input.out_port(0).connect(pow_node.in_port(0))
        div_sqrt.in_port(0).get_connection().set_destination(div_node.in_port(0))
        div_node.in_port(1).connect(pow_node.out_port(0))
        div_sqrt.out_port(0).get_connection().set_source(div_node.out_port(0))

        rename_nodes([(div_sqrt, div_sqrt_name + '/ShouldBeDeleted'), (div_node, div_sqrt_name)])
