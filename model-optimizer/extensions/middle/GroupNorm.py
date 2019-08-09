"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Dict

import numpy as np

from extensions.ops.mvn import MVN
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from extensions.ops.elementwise import Mul, Add
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_spatial_dimensions_value, node_to_get_features_dimension_value, \
    node_to_get_batch_value, new_shape_node_from_shape_nodes


class GroupNormToMVN(MiddleReplacementPattern):
    """
    Converts GroupNorm operation to Reshape + MVN + Reshape + Mul + Add
    """
    op = 'GroupNorm'
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.EltwiseChecker import EltwiseChecker
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

        # node computing initial GroupNorm input shape
        initial_shape_op_node = Shape(graph, {'name': group_norm_node.name + '/Shape'}).create_node()
        initial_shape_op_node.in_port(0).connect(group_norm_node.in_port(0).get_source())

        initial_batch_dim_node = node_to_get_batch_value(initial_shape_op_node)
        initial_features_dim_node = node_to_get_features_dimension_value(initial_shape_op_node)
        initial_spatial_dims_node = node_to_get_spatial_dimensions_value(initial_shape_op_node)
        group_size_node = Const(graph, {'value': int64_array([group_norm_node.num_groups]),
                                        'name': group_norm_node.name + '/GroupSize'}).create_node()

        # calculate "features // group_size" value
        reciprocal_group_size_node = Const(graph, {'value': np.array([1.0 / group_norm_node.num_groups]),
                                                   'name': group_norm_node.name + '/ReciprocalGroupSize'}).create_node()

        c_div_g_node = Mul(graph, {}).create_node()
        c_div_g_node.in_port(0).connect(initial_features_dim_node.out_port(0))
        c_div_g_node.in_port(1).connect(reciprocal_group_size_node.out_port(0))

        batch_mul_group_size_node = Mul(graph, {}).create_node()
        batch_mul_group_size_node.in_port(0).connect(initial_batch_dim_node.out_port(0))
        batch_mul_group_size_node.in_port(1).connect(group_size_node.out_port(0))

        # create new node which concatenates several dims to one
        new_shape_node = new_shape_node_from_shape_nodes([batch_mul_group_size_node, c_div_g_node,
                                                          initial_spatial_dims_node])

        reshape_for_mvn_node = Reshape(graph, {}).create_node()

        group_norm_node.in_port(0).get_connection().set_destination(reshape_for_mvn_node.in_port(0))
        reshape_for_mvn_node.in_port(1).connect(new_shape_node.out_port(0))

        # Reshape the gamma and beta constants to correct layout from [C] to [1,C,1,1]
        gamma_value = group_norm_node.in_port(1).get_source().data.get_value()
        beta_value = group_norm_node.in_port(2).get_source().data.get_value()
        assert gamma_value is not None, 'The gamma should be constant'
        assert beta_value is not None, 'The beta should be constant'
        gamma_value = np.reshape(gamma_value, [1, -1, 1, 1])
        group_norm_node.in_port(1).get_source().data.set_value(gamma_value)
        beta_value = np.reshape(beta_value, [1, -1, 1, 1])
        group_norm_node.in_port(2).get_source().data.set_value(beta_value)

        # MVN
        mvn_node = MVN(graph, {'name': group_norm_node.name + '/MVN',
                               'across_channels': 1,
                               'normalize_variance': 1,
                               'eps': group_norm_node.eps}).create_node()
        mvn_node.in_port(0).connect(reshape_for_mvn_node.out_port(0))

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
