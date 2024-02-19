# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.const import Const


def get_node_with_permutation(node: Node, port_info: str):
    node_type, port = port_info.split(':')
    port = int(port)
    return node.in_node(port) if node_type == 'input' else node.out_node(port)


def axis(op_node: Node, port_info: str, input_port: int):
    """
    Performs layout change related transformation of the data on the in_port_idx port of op_node.
    Translates shape indexes from one layout to another according to inverse permutation

    Transformation inserts Gather operation with
        permutation as 0-port input data and
        actual data to translate as 1-port input indexes of Gather

    For example:
        NHWC Reduce operation has 0-port input with data of shape [1, 2, 3, 4] and
        1-port input with axis indices [0, 1].

        After translating such operation to NCHW layout:
            0-port input shape = [1, 4, 2, 3]
            1-port input axis indices = [0, 2]
    """
    graph = op_node.graph

    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    data_node = op_node.in_node(input_port)

    gather_name = op_node.soft_get('name', op_node.id) + '/AxisGather'
    const = Const(graph, {'value': permutation.inv, 'name': gather_name + '/const',
                          'need_shape_inference': True}).create_node_with_data()
    axis_const = Const(graph, {'value': int64_array(0), 'name': gather_name + '/axis'}).create_node_with_data()
    gather = Gather(graph, {'name': gather_name, 'need_shape_inference': True}).create_node_with_data(
        [const, data_node, axis_const])
    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()
    graph.add_edge(gather.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)
    op_node['need_shape_inference'] = True


def order(op_node: Node, port_info: str, input_port: int):
    """
        Performs layout change related transformation of the data on the in_port_idx port of op_node.
        Translates ordered shape indexes from one layout to another according to permutation

        Transformation inserts two Gather operations

        1 Gather reorders data to new layout according to direct permutation:
            actual data to translate as 1-port input indexes of Gather and
            permutation as 0-port input data
        2 Gather translates shape indexes from one layout to another according to inverse permutation
            permutation as 0-port input data and
            actual data to translate as 1-port input indexes of Gather

    For example:
        NHWC Transpose operation has 0-port input with data of shape [1, 2, 3, 4] and
        1-port input with new order indices [0, 1, 3, 2].

        After translating such operation to NCHW layout:
            0-port input shape = [1, 4, 2, 3]

        1 phase (after first Gather insertion):
            1-port input order indices = [0, 2, 1, 3]
        2 phase (after second Gather insertion):
            1-port input order indices = [0, 3, 2, 1]
    """
    graph = op_node.graph
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    data_node = op_node.in_node(input_port)

    gather_name = op_node.soft_get('name', op_node.id) + '/OrderGather_1'
    const = Const(graph, {'value': permutation.perm, 'name': gather_name + '/const',
                          'need_shape_inference': True}).create_node_with_data()
    axis_const = Const(graph, {'value': int64_array(0), 'name': gather_name + '/axis'}).create_node_with_data()
    gather = Gather(graph, {'name': gather_name,
                            'need_shape_inference': True}).create_node_with_data([data_node, const, axis_const])

    gather_1_name = op_node.soft_get('name', op_node.id) + '/OrderGather_2'
    const_1 = Const(graph, {'value': permutation.inv, 'name': gather_1_name + '/const',
                            'need_shape_inference': True}).create_node_with_data()
    axis_const_1 = Const(graph, {'value': int64_array(0), 'name': gather_1_name + '/axis'}).create_node_with_data()
    gather_1 = Gather(graph, {'name': gather_1_name,
                              'need_shape_inference': True}).create_node_with_data([const_1, gather, axis_const_1])

    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()
    graph.add_edge(gather_1.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)
    op_node['need_shape_inference'] = True


def strided_slice(op_node: Node, port_info: str, input_port: int):
    """
    StridedSLice must be permuted even if input or output tensors have rank lesser than 4
    e.g. input_shape = (1, 10, 10), out = input[:, 0:10, :, new_axis], input_rank < 4
    input_shape = (1, 10, 10, 3), out = input[:, 0:5, 0:4, 0], output_rank < 4
    in both examples slice_rank is >= 4
    slice_rank is defined by length of begin, end, strides (they all are of the same length)
    """
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permute_indices_for_gather = permutation_data_node.permutation.perm
    if len(permute_indices_for_gather) == 0:
        return
    from openvino.tools.mo.ops.op import PermuteAttrs

    slice_rank = op_node.in_port(input_port).data.get_shape()[0]  # length of begin, end or strides
    permute_indices_for_gather = PermuteAttrs.get_nhwc_to_nchw_permutation(slice_rank).perm
    reorder_inputs_for_shape_or_slice(op_node, input_port, permute_indices_for_gather)


def shape(op_node: Node, port_info: str, input_port: int):
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permute_indices_for_gather = permutation_data_node.permutation.perm
    if len(permute_indices_for_gather) == 0:
        return
    reorder_inputs_for_shape_or_slice(op_node, input_port, permute_indices_for_gather)


def reorder_inputs_for_shape_or_slice(op_node: Node, input_port: int, permute_indices_for_gather: list):
    """
    axis and slice permutations are almost the same the only difference is that for slice in general
    case permutation depends from slice_rank not from input_rank or output_rank
    """
    graph = op_node.graph
    data_node = op_node.in_node(input_port)

    gather_name = op_node.soft_get('name', op_node.id) + '/ShapeGather'
    const = Const(graph, {'value': permute_indices_for_gather, 'name': gather_name + '/const',
                          'need_shape_inference': True}).create_node_with_data()
    axis_const = Const(graph, {'value': int64_array(0), 'name': gather_name + '/axis'}).create_node_with_data()
    gather = Gather(graph, {'name': gather_name,
                            'need_shape_inference': True}).create_node_with_data([data_node, const, axis_const])
    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()

    graph.add_edge(gather.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)

    # need to run manually to override output shape value to resolve shape collision for nodes with
    # 'correct_data_layout' output port attrs
    op_node['need_shape_inference'] = True


def transpose(op_node: Node, port_info: str, input_port: int):
    graph = op_node.graph
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), \
        'Data node "{}" does not have permutation for node {}, port_info "{}".'.format(
            permutation_data_node.id, op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    transpose_name = op_node.soft_get('name', op_node.id) + '/Transpose'
    from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs  # avoiding recursive imports
    transpose = create_op_with_const_inputs(
        graph, Transpose, {1: permutation.perm}, {'name': transpose_name, 'override_output_shape': True})
    op_node.in_port(input_port).get_connection().insert_node(transpose)
    transpose.infer(transpose)


def transpose_nchw_to_nhwc(op_node: Node, port_info: str, input_port: int):
    graph = op_node.graph
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    rank = len(permutation_data_node.shape)
    assert rank >= 4, 'Rank must be 4D or higher for HCHW to HHWC permutation on node {}.'.format(op_node.id)

    perm = list(range(rank))
    perm.insert(1, perm.pop())
    perm = int64_array(perm)

    transpose_name = op_node.soft_get('name', op_node.id) + '/Transpose'
    from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs  # avoiding recursive imports
    transpose = create_op_with_const_inputs(
        graph, Transpose, {1: perm}, {'name': transpose_name, 'override_output_shape': True})
    op_node.in_port(input_port).get_connection().insert_node(transpose)
    transpose.infer(transpose)


class PermuteInputs:
    input_permutes = {
        'axis': lambda node, port_info, input_port: axis(node, port_info, input_port),
        'slice': lambda node, port_info, input_port: strided_slice(node, port_info, input_port),
        'order': lambda node, port_info, input_port: order(node, port_info, input_port),
        'shape': lambda node, port_info, input_port: shape(node, port_info, input_port),
        'transpose': lambda node, port_info, input_port: transpose(node, port_info, input_port),
        'transpose_nchw_to_nhwc': lambda node, port_info, input_port: transpose_nchw_to_nhwc(node, port_info,
                                                                                             input_port),
    }

    shape_check_rules = {
        'rank': lambda port: bool(len(port.data.get_shape()) >= 4),
        'dim_size': lambda port: bool(port.data.get_shape()[0] >= 4),  # if input 'dim_size' >= 4 need to permute
    }

    def set_input_permutation(self, node1: Node, node2: Node, port_info: str, permutation_rule: str,
                              shape_check_rule: str = 'rank'):
        """
        Sets input permutation attribute on the edge between node1 and node2.
        Input permutation consists of function that perform input permutation and
        input port info 'input' or 'output' + <port_number> that points on the input with PermuteAttr.Permutation which
        current input depends on.

        shape_check_rule defines the check rule if the op node inputs need to be permuted.
        By default 'rank' rule is applied, 'dim_size' is used only for StridedSlice so far.
        """
        assert permutation_rule in self.input_permutes, 'No `{}` permutation rule in {}'.format(permutation_rule,
                                                                                                __class__.__name__)
        assert shape_check_rule in self.shape_check_rules, 'No `{}` permutation shape check rule ' \
                                                           'in {}'.format(shape_check_rule, __class__.__name__)
        nx.set_edge_attributes(G=node1.graph,
                               values={(node1.id, node2.id, 0): (self.input_permutes[permutation_rule], port_info,
                                                                 self.shape_check_rules[shape_check_rule])},
                               name='input_permutation')
