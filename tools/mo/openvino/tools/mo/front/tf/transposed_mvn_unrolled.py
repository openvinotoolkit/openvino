# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.PowerToEltwises import PowerToEltwises
from openvino.tools.mo.front.tf.mvn_unrolled import MVNUnrolled
from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape


def check_applicability(match: dict) -> bool:
    mean = match['mean']
    mean_reduction = mean.in_port(1).get_connection().get_source().node
    variance_reduction = match['variance'].in_port(1).get_connection().get_source().node
    pow2 = match['pow']['power']
    add = match['add']
    variance = match['variance']
    eps_port_num = 0 if add.in_port(0).get_connection().get_source().node.id != variance.id else 1
    eps = add.in_port(eps_port_num).get_connection().get_source().node

    new_name = match['division'].name + '/MVN/MVN_T_'

    if not (mean_reduction.has_valid('value') and variance_reduction.has_valid('value')):
        log.debug('Reduction indices for mean and variance for MVN node {} are not constants'.format(new_name))
        return False

    if not (all(mean_reduction.value == variance_reduction.value)):
        log.debug('Reduction indices for mean {} and variance {} do not match.'.format(
            mean_reduction.value,
            variance_reduction.value
        ))
        return False

    if not eps.has_valid('value'):
        log.debug('epsilon value for MVN node {} is not constant'.format(new_name))
        return False

    if pow2 != 0.5:
        log.debug('Power for MVN node {} ({}) is not equal to 0.5'.format(new_name, pow2))
        return False

    return True


class TransposedMVNUnrolled(FrontReplacementSubgraph):
    """
    This transformation looks for mean value normalization (across selected channels) implemented using simple
    operations and replaces found pattern with a sequence Reshape, Transpose, MVN, Transpose, Reshape, Mul, Add.

    Here we assume that
        1) the input of 'transpose' is in NHWC layout and is a 4D-tensor
        2) the constant for 'transpose' is equal to [0, 3, 1, 2]
        3) the shape for 'reshape' is [N, C1, C2, H, W]
        4) reduction indices for 'mean' and 'variance' are [2, 3, 4]
        5) the shape of 'reshape2' is equal to [N, C, H, W]
        6) the constant for 'transpose2' is [0, 2, 3, 1]

    Found pattern will be replaced with
        nodes=[
            ('new_reshape', dict(kind='op', op='Reshape')),
            ('first_permute', dict(kind='op', op='Transpose')),
            ('mvn_node', dict(kind='op', op='MVN')),
            ('second_permute', dict(kind='op', op='Transpose')),
            ('new_reshape2', dict(kind='op', op='Reshape')),
            ('new_mul', dict(kind='op', op='Mul')),
            ('new_add_2', dict(kind='op', op='Add'))
        ],
        edges=[
            ('new_reshape', 'first_permute', {'in': 0}),
            ('first_permute', 'mvn_node', {'in': 0}),
            ('mvn_node', 'second_permute', {'in': 0}),
            ('second_permute', 'new_reshape2', {'in': 0}),
            ('new_reshape2', 'new_mul', {'in': 0}),
            ('new_mul', 'new_add_2', {'in': 0}),
        ]

    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.tf.mvn import MVNReplacer
        return [MVNReplacer, MVNUnrolled, PowerToEltwises]

    def pattern(self):
        log.debug('Enabled Transposed MVN replacement')
        return dict(
            nodes=[
                ('transpose', dict(kind='op', op='Transpose')),
                ('reshape', dict(kind='op', op='Reshape')),
                ('mean', dict(kind='op', op='ReduceMean')),
                ('stop_grad', dict(kind='op', op='StopGradient')),
                ('sqdiff', dict(kind='op', op='SquaredDifference')),
                ('variance', dict(kind='op', op='ReduceMean')),
                ('add', dict(kind='op', op='Add')),
                ('pow', dict(kind='op', op='AttributedPower')),
                ('sub', dict(kind='op', op='Sub')),
                ('division', dict(kind='op', op='Div')),
                ('reshape2', dict(kind='op', op='Reshape')),
                ('reshape3', dict(kind='op', op='Reshape')),
                ('reshape4', dict(kind='op', op='Reshape')),
                ('gamma_identity', dict(kind='op', op='Identity')),
                ('mul', dict(kind='op', op='Mul')),
                ('beta_identity', dict(kind='op', op='Identity')),
                ('add2', dict(kind='op', op='Add')),
                ('transpose2', dict(kind='op', op='Transpose')),
            ],
            edges=[
                ('transpose', 'reshape'),
                ('reshape', 'mean'),
                ('reshape', 'sub', {'in': 0}),
                ('reshape', 'sqdiff', {'in': 0}),
                ('mean', 'stop_grad', {'in': 0}),
                ('stop_grad', 'sqdiff', {'in': 1}),
                ('sqdiff', 'variance', {'in': 0}),
                ('mean', 'sub', {'in': 1}),
                ('variance', 'add'),
                ('add', 'pow', {'in': 0}),
                ('pow', 'division', {'in': 1}),
                ('sub', 'division', {'in': 0}),
                ('division', 'reshape2'),
                ('reshape2', 'mul', {'in': 0}),
                ('reshape3', 'mul', {'in': 1}),
                ('gamma_identity', 'reshape3'),
                ('mul', 'add2', {'in': 0}),
                ('reshape4', 'add2', {'in': 1}),
                ('beta_identity', 'reshape4'),
                ('add2', 'transpose2'),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        if not check_applicability(match):
            return

        reshape = match['reshape']
        div_name = match['division'].name

        input_shape = Shape(graph, dict(name=div_name + '/shape/MVN_T_')).create_node()
        shape_of_reshape = reshape.in_port(1).get_connection().get_source().node.value
        c1, c2 = shape_of_reshape[1], shape_of_reshape[2]
        c = c1 * c2

        new_reshape = create_op_node_with_second_input(graph, Reshape, int64_array([0, 0, 0, c1, c2]),
                                                       dict(name=div_name + '/first_reshape/MVN_T_'))
        permute_order = int64_array([0, 1, 2, 4, 3])
        first_permute = create_op_node_with_second_input(graph, Transpose, permute_order,
                                                         dict(name=div_name + '/first_permute/MVN_T_'), new_reshape)

        add = match['add']
        variance = match['variance']
        eps_port_num = 0 if add.in_port(0).get_connection().get_source().node.id != variance.id else 1
        eps = add.in_port(eps_port_num).get_connection().get_source().node
        mvn_node = create_op_with_const_inputs(graph, MVN, {1: int64_array([1, 2, 3])},
                                               dict(name=div_name + '/MVN/MVN_T_',
                                                    eps=eps.value, normalize_variance=1,
                                                    eps_mode='inside_sqrt'))
        first_permute.out_port(0).connect(mvn_node.in_port(0))

        second_permute = create_op_node_with_second_input(graph, Transpose, permute_order,
                                                          dict(name=div_name + '/second_permute/MVN_T_'), mvn_node)
        new_reshape2 = Reshape(graph, dict(name=div_name + '/second_reshape/MVN_T_')).create_node()
        second_permute.out_port(0).connect(new_reshape2.in_port(0))
        gamma_val = np.reshape(match['gamma_identity'].in_port(0).get_connection().get_source().node.value,
                               int64_array([1, 1, 1, c]))
        new_mul = create_op_node_with_second_input(graph, Mul, gamma_val,
                                                   dict(name=match['mul'].name + '/MVN_T_'), new_reshape2)
        beta_val = np.reshape(match['beta_identity'].in_port(0).get_connection().get_source().node.value,
                              int64_array([1, 1, 1, c]))
        new_add2 = create_op_node_with_second_input(graph, Add, beta_val,
                                                    dict(name=match['add2'].name + '/MVN_T_'), new_mul)

        transpose_connection = match['transpose'].in_port(0).get_connection()
        before_transpose = transpose_connection.get_source().node
        transpose_connection.set_destination(new_reshape.in_port(0))
        input_shape.out_port(0).connect(new_reshape2.in_port(1))
        before_transpose.out_port(0).connect(input_shape.in_port(0))
        match['transpose2'].out_port(0).get_connection().set_source(new_add2.out_port(0))
