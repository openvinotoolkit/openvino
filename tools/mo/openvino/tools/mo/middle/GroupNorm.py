# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.utils.shape import node_to_get_spatial_dimensions_value, node_to_get_features_dimension_value, \
    node_to_get_batch_value, new_shape_node_from_shape_nodes, get_shape_and_rank_nodes_by_port


class GroupNormToMVN(MiddleReplacementPattern):
    """
    Converts GroupNorm operation to Reshape + MVN + Reshape + Mul + Add
    """
    op = 'GroupNorm'
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.EltwiseChecker import EltwiseChecker
        # TODO the EltwiseChecker does not work correctly for eltwises with 1D inputs
        return [EltwiseChecker]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op='GroupNorm')),
            ],
            edges=[])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        group_norm_node = match['op']
        group_norm_num_input_dims = len(group_norm_node.in_port(0).data.get_shape())

        # node computing initial GroupNorm input shape
        initial_shape_op_node = Shape(graph, {'name': group_norm_node.name + '/Shape'}).create_node()
        initial_shape_op_node.in_port(0).connect(group_norm_node.in_port(0).get_source())

        initial_shape_op_node_float = Cast(
            graph, {'name': initial_shape_op_node.name + '/to_float',
                    'dst_type': data_type_str_to_np(graph.graph['cmd_params'].data_type)}).create_node()
        initial_shape_op_node.out_port(0).connect(initial_shape_op_node_float.in_port(0))

        initial_batch_dim_node = node_to_get_batch_value(initial_shape_op_node_float)
        initial_features_dim_node = node_to_get_features_dimension_value(initial_shape_op_node_float)
        initial_spatial_dims_node_int = node_to_get_spatial_dimensions_value(initial_shape_op_node)
        initial_spatial_dims_node = Cast(
            graph, {'name': initial_spatial_dims_node_int.name + '/to_float',
                    'dst_type': data_type_str_to_np(graph.graph['cmd_params'].data_type)}).create_node()
        initial_spatial_dims_node_int.out_port(0).connect(initial_spatial_dims_node.in_port(0))

        group_size_node = Const(graph, {'value': int64_array([group_norm_node.num_groups]),
                                        'name': group_norm_node.name + '/GroupSize'}).create_node()

        # calculate "features // group_size" value
        reciprocal_group_size_node = Const(graph, {'value': mo_array([1.0 / group_norm_node.num_groups]),
                                                   'name': group_norm_node.name + '/ReciprocalGroupSize'}).create_node()

        c_div_g_node = Mul(graph, {}).create_node()
        c_div_g_node.in_port(0).connect(initial_features_dim_node.out_port(0))
        c_div_g_node.in_port(1).connect(reciprocal_group_size_node.out_port(0))

        batch_mul_group_size_node = Mul(graph, {}).create_node()
        batch_mul_group_size_node.in_port(0).connect(initial_batch_dim_node.out_port(0))
        batch_mul_group_size_node.in_port(1).connect(group_size_node.out_port(0))

        # create new node which concatenates several dims to one
        new_shape_node_float = new_shape_node_from_shape_nodes([batch_mul_group_size_node, c_div_g_node,
                                                                initial_spatial_dims_node])
        new_shape_node = Cast(graph,
                              {'name': new_shape_node_float.name + '/to_int64', 'dst_type': np.int64}).create_node()
        new_shape_node_float.out_port(0).connect(new_shape_node.in_port(0))

        reshape_for_mvn_node = Reshape(graph, {}).create_node()

        group_norm_node.in_port(0).get_connection().set_destination(reshape_for_mvn_node.in_port(0))
        reshape_for_mvn_node.in_port(1).connect(new_shape_node.out_port(0))

        # Reshape the gamma and beta constants to correct layout from [C] to [1,C], [1,C,1], [1,C,1,1] etc
        gamma_beta_shape = np.ones([group_norm_num_input_dims], dtype=np.int64)
        gamma_beta_shape[1] = -1

        gamma_value = group_norm_node.in_port(1).get_source().data.get_value()
        beta_value = group_norm_node.in_port(2).get_source().data.get_value()
        assert gamma_value is not None, 'The gamma should be constant'
        assert beta_value is not None, 'The beta should be constant'
        gamma_value = np.reshape(gamma_value, gamma_beta_shape)
        group_norm_node.in_port(1).get_source().data.set_value(gamma_value)
        beta_value = np.reshape(beta_value, gamma_beta_shape)
        group_norm_node.in_port(2).get_source().data.set_value(beta_value)

        # MVN
        mvn_node = MVN(graph, {'name': group_norm_node.name + '/MVN',
                               'normalize_variance': 1,
                               'eps': group_norm_node.eps,
                               'eps_mode': 'inside_sqrt'}).create_node()
        mvn_node.in_port(0).connect(reshape_for_mvn_node.out_port(0))

        # MVN axes
        _, rank = get_shape_and_rank_nodes_by_port(mvn_node.in_port(0).get_connection().get_source(),
                                                   return_as_a_scalar=True)
        rng = create_op_with_const_inputs(graph, Range, {0: int64_array(1), 2: int64_array(1)},
                                          {'name': group_norm_node.name + '/Range', 'output_type': np.int64})
        mvn_node.in_port(1).connect(rng.out_port(0))
        rng.in_port(1).connect(rank.out_port(0))

        # reshape to the initial shape before multiplying with gamma and adding beta
        reshape_to_initial_shape_node = Reshape(graph, {}).create_node()
        reshape_to_initial_shape_node.in_port(0).connect(mvn_node.out_port(0))
        reshape_to_initial_shape_node.in_port(1).connect(initial_shape_op_node.out_port(0))

        mul_node = Mul(graph, {'name': mvn_node.name + '/Mul'}).create_node()
        mul_node.in_port(0).connect(reshape_to_initial_shape_node.out_port(0))
        group_norm_node.in_port(1).get_connection().set_destination(mul_node.in_port(1))

        add_node = Add(graph, {'name': mul_node.name + '/Add'}).create_node()
        add_node.in_port(0).connect(mul_node.out_port(0))
        group_norm_node.in_port(2).get_connection().set_destination(add_node.in_port(1))

        group_norm_node.out_port(0).get_connection().set_source(add_node.out_port(0))
