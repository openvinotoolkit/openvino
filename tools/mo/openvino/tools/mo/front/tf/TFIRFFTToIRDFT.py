# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.ops.dft import IRDFT
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class TFIRFFTToIRDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFIRFFT into OpenVINO IRDFT.
    """
    enabled = True

    # def run_after(self):
    #     from openvino.tools.mo.front.tf.RollRealImagPack import RollRealImagPack
    #     return [RollRealImagPack]

    def find_and_replace_pattern(self, graph: Graph):
        for tf_irfft in graph.get_op_nodes(op='TFIRFFT'):
            tf_irfft_name = tf_irfft.soft_get('name', tf_irfft.id)

            num_of_dims = tf_irfft.soft_get('num_of_dimensions', 1)
            axes = int64_array(range(-num_of_dims, 0))
            irdft_node = create_op_with_const_inputs(graph, IRDFT, {1: axes}, {'in_ports_count': 2},
                                                     tf_irfft.in_port(0).get_source().node)

            tf_irfft.out_port(0).get_connection().set_source(irdft_node.out_port(0))

            rename_nodes([(tf_irfft, tf_irfft_name + '/to_be_removed'), (irdft_node, tf_irfft_name)])

            if graph.graph['layout'] == 'NHWC':
                irdft_node['need_insert_transposes_for_dft'] = True
