# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.ops.dft import RDFT
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class TFRFFTToRDFT(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFRFFT into OpenVINO RDFT.
    """
    enabled = True

    # def run_after(self):
    #     from openvino.tools.mo.front.tf.RollRealImagPack import RollRealImagPack
    #     return [RollRealImagPack]

    def find_and_replace_pattern(self, graph: Graph):
        for tf_rfft in graph.get_op_nodes(op='TFRFFT'):
            tf_rfft_name = tf_rfft.soft_get('name', tf_rfft.id)

            num_of_dims = tf_rfft.soft_get('num_of_dimensions', 1)
            axes = int64_array(range(-num_of_dims, 0))
            rdft_node = create_op_with_const_inputs(graph, RDFT, {1: axes}, {'in_ports_count': 2},
                                                    tf_rfft.in_port(0).get_source().node)

            tf_rfft.out_port(0).get_connection().set_source(rdft_node.out_port(0))

            rename_nodes([(tf_rfft, tf_rfft_name + '/to_be_removed'), (rdft_node, tf_rfft_name)])

            # if graph.graph['layout'] == 'NHWC':
            #     rdft_node['need_insert_transposes_for_dft'] = True
