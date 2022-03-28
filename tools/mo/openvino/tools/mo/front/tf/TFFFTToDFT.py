# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.dft import DFT, IDFT, IRDFT, RDFT


class TFFFTToDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFFFT into OpenVINO DFT (if the attribute 'is_inverse' is False),
    or into OpenVINO IDFT (otherwise).
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
            assert tf_fft['fft_kind'] in ['DFT', 'IDFT', 'RDFT', 'IRDFT'], \
                'Node {} with the operation TFFFT supports only the following FFT-like operations: ' \
                'DFT, IDFT, RDFT, IRDFT. Got: {}'.format(tf_fft_name, tf_fft['fft_kind'])
            op = {'DFT': DFT, 'IDFT': IDFT, 'RDFT': RDFT, 'IRDFT': IRDFT}[tf_fft['fft_kind']]
            dft_node = create_op_with_const_inputs(graph, op, {1: axes}, {'in_ports_count': 2},
                                                   tf_fft.in_port(0).get_source().node)

            tf_fft.out_port(0).get_connection().set_source(dft_node.out_port(0))

            rename_nodes([(tf_fft, tf_fft_name + '/to_be_removed'), (dft_node, tf_fft_name)])

            if graph.graph['layout'] == 'NHWC':
                dft_node['need_insert_transposes_for_dft'] = True
