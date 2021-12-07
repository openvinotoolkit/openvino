# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import numpy as np

from extensions.ops.Cast import Cast
from extensions.ops.elementwise import Div
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from mo.graph.graph import Graph, Node, Port
from mo.ops.concat import Concat
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_shape_value_of_indices


def find_max_frame_time(node: Node):
    """
    Find maximum time_dim among all inputs of given node
    If one of inputs have not set time_dim (=-1), then we can't find out time_dim for given node and return None
    """
    in_frame_time_max = 0

    for inp in node.in_ports():
        if node.in_port(inp).disconnected():
            continue
        in_node = node.in_port(inp).get_source().node
        if in_node.time_dim == -1:
            return None
        if in_node.time_dim > in_frame_time_max:
            in_frame_time_max = in_node.time_dim

    return in_frame_time_max


def propagate_time_dim_through_branch(node: Node, dest_port: Port):
    """
    Propagate time_dim for one branch starting from given node:dest_port.
    MemoryOffset/Splice nodes increase time_dim
    Convolution/Pooling decrease time_dim to 0 value because it works through the whole context
    other nodes don't change time_dim
    If we find out node with several inputs, one of which have undefined time_dim, process stops.
    """
    child = dest_port.node

    if child.time_dim >= 0:
        return

    if child.op == "MemoryOffset":
        child.time_dim = node.time_dim + abs(child.t)
    elif child.op == "Splice":
        child.time_dim = node.time_dim + len(child.context) - 1
    elif child.op in ['Convolution', 'Pooling']:
        child.time_dim = 0
    elif len(child.in_edges()) > 1:
        tmp_time_dim = find_max_frame_time(child)
        if tmp_time_dim is not None:
            child.time_dim = tmp_time_dim
        else:
            return
    else:  # len(child.in_edges()) == 1
        child.time_dim = node.time_dim

    for p in child.out_ports():
        if child.out_port(p).disconnected():
            continue
        for dest in child.out_port(p).get_destinations():
            propagate_time_dim_through_branch(child, dest)


def set_time_dim(graph):
    """
    Set value of dimension where we gather frames with different time labels.
    If in some node we don't use any context, then time_dim = 0
    """
    nx.set_node_attributes(G=graph, name='time_dim', values=-1)
    start_nodes = graph.get_op_nodes(op="Const")
    start_nodes.extend(graph.get_op_nodes(op='Parameter'))
    for n in start_nodes:
        n.time_dim = 0

    for node in start_nodes:
        for p in node.out_ports():
            if node.out_port(p).disconnected():
                continue
            for dest in node.out_port(p).get_destinations():
                propagate_time_dim_through_branch(node, dest)


class ReplaceConvolutionReshape(FrontReplacementPattern):
    """
       This pass adds Reshapes and Transposes around a Convolution/Pooling layer for reshaping from NH to NCHW
       For example:
           Let's suppose we have next graph:

           Prev_Layer [N, H] -> Convolution [N, C, H, W] -> Next_Layer [N, H]

           In this case Convolution takes only [N, H] from input tensor in 3rd dim
           So this pass will convert this graph to the next one:

           Prev_Layer [N, H] -> Reshape(N, H, W, C) -> Transpose(0, 3, 1, 2) -> Convolution [N, C, H, W] ->
           Transpose(0, 2, 3, 1) -> Reshape(0, -1) -> Next_Layer [N, H]
   """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == "kaldi"]

    def run_before(self):
        from extensions.front.MatMul_normalizer import FullyConnectedDecomposer
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [SplitRecurrentMemoryOffset, FullyConnectedDecomposer]

    def run_after(self):
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        return [MemoryOffsetAdjustment]

    @staticmethod
    def find_and_replace_pattern(graph: Graph):
        conv_pool_nodes = graph.get_op_nodes(op='Convolution')
        conv_pool_nodes.extend(graph.get_op_nodes(op='Pooling'))

        if len(conv_pool_nodes) == 0:
            return

        set_time_dim(graph)

        for node in conv_pool_nodes:
            node_name = node.soft_get('name', node.id)

            # create Reshape before convolution
            # shape = [in_shape[0], t, patch_stride, C= in_shape[1]/(patch_stride*t)]
            # or before pooling
            # shape = [in_shape[0], t, in_shape[1]/(pool_stride*t), pool_stride]
            time_dim = node.in_port(0).get_source().node.time_dim + 1
            if node.op == 'Convolution':
                frame_height = node.patch_stride if node.has_valid('patch_stride') else node.height_in
                # set time t instead of 1 in kernel as H and update C to have kernel shape correct
                node.kernel[2] = time_dim
                assert node.kernel[1] % time_dim == 0
                node.kernel[1] = node.kernel[1] // time_dim
                node.kernel_spatial = node.kernel[2:]
                index_const = 2
                index_div = 3
            else:
                frame_height = node.pool_stride
                if node.pool_step is None:
                    node.stride = int64_array([1, 1, node.window[-1], node.window[-1]])
                index_const = 3
                index_div = 2

            i_shape = Shape(graph, {'name': node_name + '/Shape'}).create_node()
            i_shape.in_port(0).connect(node.in_port(0).get_source())

            N, H = node_to_get_shape_value_of_indices(i_shape, [0]), node_to_get_shape_value_of_indices(i_shape, [1])

            div = create_op_with_const_inputs(
                graph, Div, {1: int64_array([frame_height * time_dim])}, {'name': node_name + '/div_stride_h'})
            div.in_port(0).connect(H.out_port(0))

            concat = create_op_with_const_inputs(graph, Concat, {index_const: int64_array([frame_height]),
                                                                 1: int64_array([time_dim])},
                                                 {'name': node_name + '/concat_all_dims', 'in_ports_count': 4,
                                                  'axis': 0})
            concat.in_port(0).connect(N.out_port(0))
            concat.in_port(index_div).connect(div.out_port(0))

            reshape_in = Reshape(graph, {'name': node_name + '/reshape_in',
                                         'time_dim': time_dim - 1}).create_node()
            reshape_in.in_port(1).connect(concat.out_port(0))

            # change layout from NHWC to NCHW
            # should be replaced by common Permute logic in future
            direct_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
                                                                {'name': node.name + '/Transpose',
                                                                'time_dime': time_dim - 1}, reshape_in)
            # after convolution/pooling time_dim becomes 0
            inverse_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 2, 3, 1]),
                                                                 {'name': node.name + '/Transpose_back',
                                                                  'time_dim': 0})

            # create Reshape after Convolution
            reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                           {'name': node_name + '/reshape_out', 'time_dim': 0})

            # connect input_reshape_node
            source = node.in_port(0).get_source()
            node.in_port(0).get_connection().set_source(direct_transpose.out_port(0))
            reshape_in.in_port(0).connect(source)
            # connect output_reshape_node
            node.out_port(0).get_connection().set_source(reshape_out.out_port(0))
            node.out_port(0).connect(inverse_transpose.in_port(0))
            reshape_out.in_port(0).connect(inverse_transpose.out_port(0))

        for node in graph.get_op_nodes():
            if 'time_dim' in node:
                del node['time_dim']
