# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import networkx as nx

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
    child = dest_port.node

    if child.time_dim >= 0:
        return

    if child.op == "MemoryOffset":
        child.time_dim = node.time_dim + abs(child.t)
    elif child.op == "Splice":
        child.time_dim = node.time_dim + len(child.context) - 1
    elif child.op in ['Convolution', 'MaxPooling']:
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


class FindTimeLabel(FrontReplacementPattern):
    """
    Pass used to fix wrong results in the following situation:
                              input
                              |   \
                            ...   ...
                             |       \
                    MemoryOffset(k)   \
                             |        |
                             ...      |
                              \      |
                               \     |
                               Concat
    In Left branch we have MemoryOffset with k > 0 so we wait until kth frame will be calculated. In right branch
    we have no such offsets. As result we Concat (or use in any calculations with more than 1 input) kth frame from
    left branch and 0th from right branch. So we need to add synchronization before Concat node. it can be done with
    MemoryOffset(k) inserted before Concat.

    Main idea of this change that when we found memoryOffset with t>0 we should re-calculate all delays relative to this
    t.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    def run_before(self):
        from extensions.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [SplitRecurrentMemoryOffset]

    def run_after(self):
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        return [MemoryOffsetAdjustment]

    def find_and_replace_pattern(self, graph: Graph):
        should_continue = False
        convs = graph.get_op_nodes(op='Convolution')
        pools = graph.get_op_nodes(op='MaxPooling')
        if (convs is None or len(convs) == 0) and (pools is None or len(pools) == 0):
            return

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
       This pass adds Reshapes around a Convolution layer for reshaping from NH to NCHW
       For example:
           Let's suppose we have next graph:

           Prev_Layer [N, H] -> Convolution [N, C, H, W] -> Next_Layer [N, H]

           In this case Convolution takes only [N, H] from input tensor in 3rd dim
           So this pass will convert this graph to the next one:

           Prev_Layer [N, H] -> Reshape -> Convolution [N, C, H, W] -> Reshape -> Next_Layer [N, H]

   """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == "kaldi"]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        return [ReplaceConvolutionTranspose]

    def run_after(self):
        return [FindTimeLabel]

    def pattern(self):
        return dict(nodes=[('conv', dict(op='Convolution'))],
                    edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['conv']
        node_name = node.soft_get('name', node.id)

        # create Reshape before convolution
        # if transpose will be applied (new models)
        #   shape = [in_shape[0], t, patch_stride, C= in_shape[1]/(patch_stride*t)]
        # else (for old models to avoid fails on GNA - should be removed as soon as GNA will be changed)
        #   shape = [in_shape[0], t, C= in_shape[1]/(patch_stride*t), patch_stride]
        sp_dim_1 = 1
        time_dim = node.in_port(0).get_source().node.time_dim + 1
        if node.has_valid('patch_stride'):
            channel_dim = 3
            sp_dim_2 = 2
            frame_height = node.patch_stride
        else:
            channel_dim = 3
            sp_dim_2 = 2
            frame_height = node.height_in

        # set time t instead of 1 in kernel as H and update C to have kernel shape correct
        node.kernel[2] = time_dim
        assert node.kernel[1] % time_dim == 0
        node.kernel[1] = node.kernel[1] // time_dim
        node.kernel_spatial = node.kernel[2:]

        i_shape = Shape(graph, {'name': node_name + '/Shape'}).create_node()
        i_shape.in_port(0).connect(node.in_port(0).get_source())

        N, H = node_to_get_shape_value_of_indices(i_shape, [0]), node_to_get_shape_value_of_indices(i_shape, [1])

        div = create_op_with_const_inputs(
            graph, Div, {1: int64_array([frame_height * time_dim])}, {'name': node_name + '/div_stride_h'})
        div.in_port(0).connect(H.out_port(0))

        concat = create_op_with_const_inputs(graph, Concat, {sp_dim_2: int64_array([frame_height]),
                                                             sp_dim_1: int64_array([time_dim])},
                                             {'name': node_name + '/concat_all_dims', 'in_ports_count': 4, 'axis': 0})
        concat.in_port(0).connect(N.out_port(0))
        concat.in_port(channel_dim).connect(div.out_port(0))

        reshape_in = Reshape(graph, {'name': node_name + '/reshape_in'}).create_node()
        reshape_in.in_port(1).connect(concat.out_port(0))

        # change layout from NHWC to NCHW
        # should be replaced by common Permute logic in future
        direct_transpose = None
        inverse_transpose = None
        if channel_dim == 3 and node.channel_dims == 1:
            direct_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
                                                               {'name': node.name + '/Transpose'}, reshape_in)
            inverse_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 2, 3, 1]),
                                                                 {'name': node.name + '/Transpose_back'})

        # create Reshape after Convolution
        reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                       {'name': node_name + '/reshape_out'})

        # connect input_reshape_node
        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(direct_transpose.out_port(0) if direct_transpose else reshape_in.out_port(0))
        reshape_in.in_port(0).connect(source)
        # connect output_reshape_node
        node.out_port(0).get_connection().set_source(reshape_out.out_port(0))
        node.out_port(0).connect(inverse_transpose.in_port(0) if inverse_transpose else reshape_out.in_port(0))
        if inverse_transpose:
            reshape_out.in_port(0).connect(inverse_transpose.out_port(0))
