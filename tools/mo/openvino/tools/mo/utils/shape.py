# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.squeeze import Squeeze


def get_canonical_axis_index_node(rank: Node, axis: int) -> Node:
    """
    Returns positive axis value

    :param rank: the node of 0D output shape to get rank of tensor from
    :param axis: integer value from [-rank; rank - 1]
    :return: node producing positive integer value of axis
    """
    graph = rank.graph
    name = rank.soft_get('name', rank.id)
    if axis < 0:
        axis = Const(graph, {'name': name + '/negative_axis', 'value': int64_array(axis)}).create_node()
        add = Add(graph, {'name': name + '/positive_axis'}).create_node()
        rank.out_port(0).connect(add.in_port(0))
        axis.out_port(0).connect(add.in_port(1))
        return add
    else:
        return Const(graph, {'name': name + '/positive_axis', 'value': int64_array(axis)}).create_node()


def get_range_node_of_idxs(rank: Node, begin: int, end: int,
                           include_begin: bool = True, include_end: bool = False) -> Node:
    """
    Returns node that produces 1D output of values of range from begin to end (ex)/(in)cluding begin or end point

    :param rank: the node of 0D output shape to get rank of tensor from
    :param begin: integer value from [-rank; rank - 1]
    :param end: integer value from [-rank; +rank]
    :param include_begin: boolean flag to include or exclude start point from range output
    :param include_end: boolean flag to include or exclude end point from range output
    :return: range node producing 1D output
    """
    graph = rank.graph
    name = rank.soft_get('name', rank.id)

    start_idx = get_canonical_axis_index_node(rank, begin)
    end_idx = get_canonical_axis_index_node(rank, end)

    if not include_begin:
        const = Const(graph, {'value': int64_array(1), 'name': name + '/exclude_begin/value'}).create_node()
        add = Add(graph, {'name': name + '/exclude_begin'}).create_node()
        start_idx.out_port(0).connect(add.in_port(0))
        const.out_port(0).connect(add.in_port(1))
        start_idx = add

    if include_end:
        const = Const(graph, {'value': int64_array(1), 'name': name + '/including_end/value'}).create_node()
        add = Add(graph, {'name': name + '/including_end'}).create_node()
        end_idx.out_port(0).connect(add.in_port(0))
        const.out_port(0).connect(add.in_port(1))
        end_idx = add

    delta = Const(graph, {'name': name + '/delta', 'value': int64_array(1)}).create_node()
    range_node = Range(graph, {'name': name + '/range_idxs'}).create_node()

    start_idx.out_port(0).connect(range_node.in_port(0))
    end_idx.out_port(0).connect(range_node.in_port(1))
    delta.out_port(0).connect(range_node.in_port(2))

    return range_node


def get_shape_values_by_indices_node(shape_node: Node, indices_node: Node) -> Node:
    """
    The function returns a node that produces values of the specified indices node of the input node 'shape_node'

    :param shape_node: the node of 1D output shape to get elements from
    :param indices_node: the node of 1D output shape with the list of element indices to get
    :return: node producing required elements of the node
    """
    graph = shape_node.graph
    axis = Const(graph, {'value': int64_array(0), 'name': shape_node.name + '/Axis'}).create_node()
    gather_node = Gather(graph, {'name': shape_node.name + '/Gather'}).create_node()

    shape_node.out_port(0).connect(gather_node.in_port(0))
    indices_node.out_port(0).connect(gather_node.in_port(1))
    axis.out_port(0).connect(gather_node.in_port(2))
    return gather_node


def node_to_get_shape_value_of_indices(shape_node: Node, indices: list) -> Node:
    """
    The function returns a node that produces values of the specified indices of the input node 'shape_node'

    :param shape_node: the node of 1D output shape to get elements from
    :param indices: the list of element indices to get
    :return: node producing required elements of the node
    """
    graph = shape_node.graph
    indices_node = Const(graph, {'value': int64_array(indices), 'name': shape_node.name + '/Indices'}).create_node()

    gather_node = get_shape_values_by_indices_node(shape_node, indices_node)
    return gather_node


def get_shape_values_by_range_idxs(shape: Node, rank: Node, begin: int, end: int,
                                   include_begin: bool = True, include_end: bool = False):
    """
    Gathers shape values that are represented by range from begin to end (in)/(ex)cluding begin or end point

    :param shape: the node of 1D output shape to get elements from
    :param rank: the node of 0D output shape to get rank of tensor from
    :param begin: integer value from [-rank; rank - 1]
    :param end: integer value from [-rank; +rank]
    :param include_begin: boolean flag to include or exclude start point from range output
    :param include_end: boolean flag to include or exclude end point from range output
    :return: gather node producing 1D output
    """
    range_node = get_range_node_of_idxs(rank, begin, end, include_begin=include_begin, include_end=include_end)
    return get_shape_values_by_indices_node(shape, range_node)


def node_to_get_batch_value(shape_node: Node) -> Node:
    """
    The function returns a node that produces the batch value which is usually the element of the shape with index 0
    :param shape_node: the node of 1D output shape to get batch from
    :return: the node producing batch value
    """
    return node_to_get_shape_value_of_indices(shape_node, [0])


def node_to_get_features_dimension_value(shape_node: Node) -> Node:
    """
    The function returns a node that produces the feature dimension value
    :param shape_node: the node of 1D output shape to get the feature dimension value from
    :return: the node producing feature dimension value
    """
    layout = shape_node.graph.graph['layout']
    if layout == 'NCHW':
        return node_to_get_shape_value_of_indices(shape_node, [1])
    elif layout == 'NHWC':
        return node_to_get_shape_value_of_indices(shape_node, [-1])
    else:
        assert 'Unsupported layout "{}"'.format(layout)


def node_to_get_spatial_dimensions_value(shape_node: Node) -> Node:
    """
    The function returns a node that produces the spatial dimension values
    :param shape_node: the node of 1D output shape to get the spatial dimension values from
    :return: the node producing the spatial dimension values
    """
    layout = shape_node.graph.graph['layout']
    shape = shape_node.in_port(0).get_connection().get_source().data.get_shape()
    assert shape is not None, 'The shape must be inferred before running this function'

    if layout == 'NCHW':
        return node_to_get_shape_value_of_indices(shape_node, list(range(2, len(shape))))
    elif layout == 'NHWC':
        return node_to_get_shape_value_of_indices(shape_node, list(range(1, len(shape) - 1)))
    else:
        assert 'Unsupported layout "{}"'.format(layout)


def new_shape_node_from_shape_nodes(input_shape_nodes: list):
    """
    The function returns a node producing 1D tensor with concatenated shapes produced by nodes from "input_shape_nodes"
    :param input_shape_nodes: list of nodes producing 1D tensors
    :return: the node producing concatenated values of nodes from the "input_shape_nodes"
    """
    assert len(input_shape_nodes) > 0, 'The list of input shape nodes should be non-empty'
    new_shape_node = Concat(input_shape_nodes[0].graph,
                            {'axis': 0,
                             'name': input_shape_nodes[0].soft_get('name', input_shape_nodes[0].id) + '/shapes_concat'}
                            ).create_node()

    for ind, input_node in enumerate(input_shape_nodes):
        new_shape_node.add_input_port(ind)
        new_shape_node.in_port(ind).connect(input_node.out_port(0))
    return new_shape_node


def get_shape_and_rank_nodes_by_port(port: Port, return_as_a_scalar: bool = True):
    """
    The function returns nodes producing shape and rank of the data from the desired port in order to use those
    operations on the middle/back phase
    :param port: Port object that specifies node output port
    :param return_as_a_scalar: boolean flag to return 1D or 0D rank
    :return: shape and rank nodes
    """
    input_node_name = port.node.soft_get('name', port.node.id)
    graph = port.node.graph

    shape = Shape(graph, dict(name=input_node_name + '/ShapeOf')).create_node()
    rank_1_d = Shape(graph, dict(name=input_node_name + '/1dRankOf')).create_node()
    rank_1_d.in_port(0).connect(shape.out_port(0))
    shape.in_port(0).connect(port)
    if not return_as_a_scalar:
        return shape, rank_1_d

    rank = create_op_node_with_second_input(graph, Squeeze, int64_array([0]), {'name': input_node_name + '/0dRankOf'},
                                            rank_1_d)
    return shape, rank

