# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.dft import DFT, IDFT
from openvino.tools.mo.ops.elementwise import Add, Sub
from openvino.tools.mo.ops.rank import Rank
from openvino.tools.mo.ops.scatter import ScatterUpdate
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.pad import Pad
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class MXFFTToDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation MXFFT into OpenVINO DFT (if the attribute 'is_inverse' is False),
    or into OpenVINO IDFT (otherwise).

    According to https://mxnet.apache.org/versions/1.0.0/api/python/symbol/contrib.html#mxnet.symbol.contrib.fft,
    MxNet operation FFT accept 2 input data shapes: [N, d] or [N_1, N_2, N_3, d], data can only be real numbers.
    The output data has shape: [N, 2*d] or [N_1, N_2, N_3, 2*d]. The format is: [real0, imag0, real1, imag1, ...].

    Next, MxNet operation IFFT accept 2 input data shapes: [N, d] or [N_1, N_2, N_3, d]. Data is in format:
    [real0, imag0, real1, imag1, ...]. Last dimension must be an even number. The output data has shape: [N, d/2] or
    [N_1, N_2, N_3, d/2]. It is only the real part of the result.

    But OpenVINO DFT and IDFT operations uses complex input data represented as real tensors of the shape
    [N_1, ..., N_r, 2]. Also, the result of OpenVINO DFT and IDFT operations is always complex but represented as
    a real tensor of the shape [M_1, ..., M_r, 2]. If OpenVINO DFT or IDFT have no input signal_size, the output shape
    and the input shape are the same.

    Hence, to convert MxNet FFT to OpenVINO DFT, we need
    1) to convert input data from the shape [N, d] or [N_1, N_2, N_3, d] to shape [N, d, 1] or [N_1, N_2, N_3, d, 1]
       respectively;
    2) to pad converted data using pads_begin = [0, 0, 0] and pads_end = [0, 0, 1] for MxNet FFT input shape [N, d], or
       using pads_begin [0, 0, 0, 0, 0] and pads_end = [0, 0, 0, 0, 1] for MxNet FFT input shape [N_1, N_2, N_3, d],
       with mode=constant;
    3) to put padded data into DFT input 0, using (-1) in 'axes' input;
    4) to reshape calculated DFT output to the shape [N, 2 * d] for for MxNet FFT input shape [N, d], or to the shape
       [N_1, N_2, N_3, 2 * d]

    Finally, to convert MxNet IFFT to OpenVINO IDFT, we need
    1) to reshape input data from the shape [N, d] or [N_1, N_2, N_3, d] to shape [N, d // 2, 2] or
       [N_1, N_2, N_3, d // 2, 2] respectively;
    2) to put reshaped input data to the input 0 of IDFT, using (-1) in 'axes' input;
    3) to get real parts using Split + Squeeze.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for mx_fft in graph.get_op_nodes(op='MXFFT'):
            if mx_fft.soft_get('is_inverse', False):
                self.convert_ifft_to_dft(graph, mx_fft)
            else:
                self.convert_fft_to_dft(graph, mx_fft)

    def convert_fft_to_dft(self, graph: Graph, mx_fft: Node):
        mx_fft_name = mx_fft.soft_get('name', mx_fft.id)
        unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array([-1])},
                                                     {'name': mx_fft_name + '/Unsqueeze'})
        rank_node = Rank(graph, {'name': mx_fft_name + '/Rank'}).create_node()

        mx_fft_connection = mx_fft.in_port(0).get_connection()
        mx_fft_connection.set_destination(unsqueeze_node.in_port(0))
        mx_fft_connection.get_source().connect(rank_node.in_port(0))

        add_node = create_op_with_const_inputs(graph, Add, {1: int64_array(1)},
                                               {'name': mx_fft_name + '/Add'}, rank_node)
        broadcast_node1 = create_op_with_const_inputs(graph, Broadcast, {0: int64_array(0)},
                                                         {'name': mx_fft_name + '/Pad_broadcast'})
        add_node.out_port(0).connect(broadcast_node1.in_port(1))

        scatter_node = create_op_with_const_inputs(graph, ScatterUpdate,
                                                   {2: int64_array(1), 3: int64_array(0)},
                                                   {'name': mx_fft_name + '/ScatterUpdate'})
        broadcast_node1.out_port(0).connect(scatter_node.in_port(0))
        rank_node.out_port(0).connect(scatter_node.in_port(1))

        pad_node = Pad(graph, {'name': mx_fft_name + '/Pad', 'mode': 'constant'}).create_node([unsqueeze_node,
                                                                                               broadcast_node1,
                                                                                               scatter_node])

        dft_node = create_op_with_const_inputs(graph, DFT, {1: int64_array([-1])},
                                               {'name': mx_fft_name + '/DFT', 'in_ports_count': 2},
                                               pad_node)

        sub_node = create_op_with_const_inputs(graph, Sub, {1: int64_array(1)}, {'name': mx_fft_name + '/Sub'})
        rank_node.out_port(0).connect(sub_node.in_port(0))
        broadcast_node2 = create_op_with_const_inputs(graph, Broadcast, {0: int64_array(0)},
                                                      {'name': mx_fft_name + '/Reshape_broadcast'})
        sub_node.out_port(0).connect(broadcast_node2.in_port(1))
        concat_node = create_op_with_const_inputs(graph, Concat, {1: int64_array([-1, 2])},
                                                  {'name': mx_fft_name + '/New_shape', 'in_ports_count': 2, 'axis': 0},
                                                  broadcast_node2)

        reshape_node = Reshape(graph, {}).create_node([dft_node, concat_node])

        mx_fft.out_port(0).get_connection().set_source(reshape_node.out_port(0))
        rename_nodes([(mx_fft, mx_fft_name + '/to_be_removed'), (reshape_node, mx_fft_name)])

    def convert_ifft_to_dft(self, graph: Graph, mx_fft: Node):
        mx_fft_name = mx_fft.soft_get('name', mx_fft.id)

        rank_node = Rank(graph, {'name': mx_fft_name + '/rank'}).create_node()
        sub_node = create_op_with_const_inputs(graph, Sub, {1: int64_array(1)}, {'name': mx_fft_name + '/Sub'})
        rank_node.out_port(0).connect(sub_node.in_port(0))
        broadcast_node0 = create_op_with_const_inputs(graph, Broadcast, {0: int64_array(0)},
                                                      {'name': mx_fft_name + '/broadcast'})
        sub_node.out_port(0).connect(broadcast_node0.in_port(1))
        concat_node = create_op_with_const_inputs(graph, Concat, {1: int64_array([-1, 2])},
                                                  {'name': mx_fft_name + '/new_shape', 'in_ports_count': 2, 'axis': 0},
                                                  broadcast_node0)

        reshape_node = Reshape(graph, {'name': mx_fft_name + '/reshape'}).create_node()
        concat_node.out_port(0).connect(reshape_node.in_port(1))

        mx_fft_connection = mx_fft.in_port(0).get_connection()
        mx_fft_connection.set_destination(reshape_node.in_port(0))
        mx_fft_connection.get_source().connect(rank_node.in_port(0))

        dft_node = create_op_with_const_inputs(graph, IDFT, {1: int64_array([-1])},
                                               {'name': mx_fft_name + '/idft', 'in_ports_count': 2},
                                               reshape_node)

        split_node = create_op_with_const_inputs(graph, Split, {1: int64_array(-1)},
                                                 {'name': mx_fft_name + '/split', 'num_splits': 2},
                                                 dft_node)
        squeeze_node = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([-1])}, {}, split_node)

        mx_fft.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
        rename_nodes([(mx_fft, mx_fft_name + '/to_be_removed'), (squeeze_node, mx_fft_name)])
