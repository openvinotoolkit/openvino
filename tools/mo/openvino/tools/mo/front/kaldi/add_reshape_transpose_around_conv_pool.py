# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.elementwise import Div
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices


def find_max_frame_time(node: Node):
    """
    Find maximum time_dim among all inputs of given node
    time_dim can be > 0 or < 0, we will find min value (<=0) and max value (>=0) of inputs,
    time_dim for node with such inputs will max-min
    If one of inputs has undefined time_dim, it raises Error because we assume that all parent nodes already have
    set time_dim.
    """
    in_frame_time_max = 0
    in_frame_time_min = 0

    for inp in node.in_ports():
        if node.in_port(inp).disconnected():
            continue
        in_node = node.in_port(inp).get_source().node
        if in_node.time_dim is None:
            raise Error("Parent node {} does not have set time_dim".format(in_node.id))
        if in_node.time_dim > in_frame_time_max:
            in_frame_time_max = in_node.time_dim
        if in_node.time_dim < in_frame_time_min:
            in_frame_time_min = in_node.time_dim

    return in_frame_time_max - in_frame_time_min


def set_time_dim(graph):
    """
    Set value of dimension where we gather frames with different time labels.
    If in some node we don't use any context, then time_dim = 0
    """
    graph.set_node_attributes('time_dim', None)

    # set correct time dim for start Convolutions
    update_time_dim_for_start_convolution(graph)

    sorted_nodes = graph.topological_sort()

    for node in sorted_nodes:
        if node.time_dim is not None:
            continue

        if node.op == "MemoryOffset":
            # MemoryOffset can be splitted already and can be without input, time_dim = t in this case
            node.time_dim = node.in_port(0).get_source().node.time_dim + node.t if not node.in_port(0).disconnected() else node.t
        elif node.op == "Splice":
            node.time_dim = node.in_port(0).get_source().node.time_dim + len(node.context) - 1
        elif node.op in ['Convolution', 'Pooling']:
            node.time_dim = 0
        elif len([port for port in node.in_ports().values() if not port.disconnected()]) > 1:
            node.time_dim = find_max_frame_time(node)
        elif len([port for port in node.in_ports().values() if not port.disconnected()]) == 1:
            node.time_dim = node.in_port(0).get_source().node.time_dim
        else:
            node.time_dim = 0


def update_time_dim_for_start_convolution(graph):
    """
    If we have pattern like Parameter->Convolution->... then input already spliced outside network. So from set_time_dim
    time_dim will be set as 1 and it will be wrong. For such pattern time_dim should be set as kernel[1]
    (convolution run through the whole splice)
    """
    params = graph.get_op_nodes(op="Parameter")
    for param_node in params:
        for dest in param_node.out_port(0).get_destinations():
            if dest.node.op == 'Convolution':
                conv_node = dest.node
                assert param_node.time_dim is None or \
                    param_node.time_dim == conv_node.soft_get('kernel')[1] - 1, \
                    "Kaldi model have 2 Convolutions after Parameter with different time dimensions"
                # time_dim starts from 0, kernel from 1
                param_node.time_dim = conv_node.soft_get('kernel')[1] - 1


class AddReshapeTransposeAroundConvPool(FrontReplacementPattern):
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

    def run_after(self):
        # remove cycles before this transformation because topological_sort call
        from openvino.tools.mo.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [SplitRecurrentMemoryOffset]

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
            # shape = [in_shape[0], t, patch_stride, C = in_shape[1]/(patch_stride*t)],
            # where patch_stride is attribute in Convolution taken from Kaldi
            # or before pooling
            # shape = [in_shape[0], t, in_shape[1]/(pool_stride*t), pool_stride]
            # where pool_stride is attribute in Pooling taken from Kaldi
            # adapt time_dim to use in kernel as dimension
            time_dim = node.in_port(0).get_source().node.time_dim + 1
            if node.op == 'Convolution':
                frame_height = node.patch_stride if node.has_valid('patch_stride') else node.height_in
                # set time t instead of 1 in kernel as H and update C to have kernel shape
                if node.kernel[2] != time_dim:
                    assert node.kernel[2] == 1
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

            # H / (patch_stride * t)
            H_div_stride_t = create_op_with_const_inputs(
                graph, Div, {1: int64_array([frame_height * time_dim])}, {'name': node_name + '/div_stride_h'})
            H_div_stride_t.in_port(0).connect(H.out_port(0))

            # gather all dims
            concat_dims = create_op_with_const_inputs(graph, Concat, {index_const: int64_array([frame_height]),
                                                                      1: int64_array([time_dim])},
                                                      {'name': node_name + '/concat_all_dims', 'in_ports_count': 4,
                                                       'axis': 0})
            concat_dims.in_port(0).connect(N.out_port(0))
            concat_dims.in_port(index_div).connect(H_div_stride_t.out_port(0))

            reshape_in = Reshape(graph, {'name': node_name + '/reshape_in',
                                         'time_dim': time_dim - 1}).create_node()
            reshape_in.in_port(1).connect(concat_dims.out_port(0))

            # change layout from NHWC to NCHW
            # should be replaced by common Permute logic in future
            direct_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 3, 1, 2]),
                                                                {'name': node_name + '/Transpose',
                                                                'time_dime': time_dim - 1}, reshape_in)
            # after convolution/pooling time_dim becomes 0
            inverse_transpose = create_op_node_with_second_input(graph, Transpose, int64_array([0, 2, 3, 1]),
                                                                 {'name': node_name + '/Transpose_back',
                                                                  'time_dim': 0})

            # create Reshape after Convolution
            reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                           {'name': node_name + '/reshape_out',
                                                            'special_zero': True, 'time_dim': 0})

            # connect input_reshape_node
            source = node.in_port(0).get_source()
            node.in_port(0).get_connection().set_source(direct_transpose.out_port(0))
            reshape_in.in_port(0).connect(source)
            # connect output_reshape_node
            node.out_port(0).get_connection().set_source(reshape_out.out_port(0))
            node.out_port(0).connect(inverse_transpose.in_port(0))
            reshape_out.in_port(0).connect(inverse_transpose.out_port(0))
            rename_nodes([(node, node_name + '/' + node.op), (reshape_out, node_name)])

        for node in graph.get_op_nodes():
            if 'time_dim' in node:
                del node['time_dim']
