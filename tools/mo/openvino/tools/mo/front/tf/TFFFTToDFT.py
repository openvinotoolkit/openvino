# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.dft import DFT, IDFT, IRDFT, RDFT


class TFFFTToDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFFFT into OpenVINO operations DFT, RDFT, IDFT, or IRDFT,
    according to the following rules:
        1) FFT, FFT2D, FFT3D are converted into DFT;
        2) IFFT, IFFT2D, IFFT3D are converted into IDFT;
        3) RFFT, RFFT2D, RFFT3D are converted into RDFT;
        4) IRFFT, IRFFT2D, IRFFT3D are converted into IRDFT.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.tf.RollRealImagPack import RollRealImagPack
        return [RollRealImagPack]

    def find_and_replace_pattern(self, graph: Graph):
        for tf_fft in graph.get_op_nodes(op='TFFFT'):
            tf_fft_name = tf_fft.soft_get('name', tf_fft.id)

            num_of_dims = tf_fft.soft_get('num_of_dimensions', 1)
            axes = int64_array(range(-num_of_dims, 0))

            fft_kind = tf_fft['fft_kind']
            assert fft_kind in ['DFT', 'IDFT', 'RDFT', 'IRDFT'], \
                'Node {} with the operation TFFFT supports only the following FFT-like operations: ' \
                'DFT, IDFT, RDFT, IRDFT. Got: {}'.format(tf_fft_name, fft_kind)

            op = {'DFT': DFT, 'IDFT': IDFT, 'RDFT': RDFT, 'IRDFT': IRDFT}[fft_kind]

            if fft_kind in ['DFT', 'IDFT'] or not tf_fft.is_in_port_connected(1):
                dft_node = create_op_with_const_inputs(graph, op, {1: axes}, {'in_ports_count': 2},
                                                       tf_fft.in_port(0).get_source().node)
            else:
                dft_node = create_op_with_const_inputs(graph, op, {1: axes}, {'in_ports_count': 3},
                                                       tf_fft.in_port(0).get_source().node)
                tf_fft.in_port(1).get_source().connect(dft_node.in_port(2))

            tf_fft.out_port(0).get_connection().set_source(dft_node.out_port(0))

            rename_nodes([(tf_fft, tf_fft_name + '/to_be_removed'), (dft_node, tf_fft_name)])

            if graph.graph['layout'] == 'NHWC':
                dft_node['need_insert_transposes_for_dft'] = True
