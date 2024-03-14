# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.ops.squeeze import Squeeze


class RFFTRealImagToRDFTSplit(FrontReplacementSubgraph):
    """
    This transformation converts the operation TFRFFT into OpenVINO RDFT.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.tf.TFFFTToDFT import TFFFTToDFT
        return [TFFFTToDFT]

    def pattern(self):
        return dict(
            nodes=[
                ('rfft', dict(op='TFFFT', fft_kind='RDFT')),
                ('real', dict(op='Real')),
                ('imag', dict(op='Imag')),
            ],
            edges=[
                ('rfft', 'real', {'in': 0}),
                ('rfft', 'imag', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        rfft_node = match['rfft']
        real_node = match['real']
        imag_node = match['imag']

        rfft_name = rfft_node.soft_get('name', rfft_node.id)
        real_name = rfft_node.soft_get('name', real_node.id)
        imag_name = rfft_node.soft_get('name', imag_node.id)
        split_node = create_op_with_const_inputs(graph, Split, {1: int64_array(-1)},
                                                 {
                                                     'name': rfft_name + '/split',
                                                     'num_splits': 2,
                                                     'out_ports_count': 2
                                                 })
        squeeze_real = create_op_with_const_inputs(graph, Squeeze, {1: int64_array(-1)},
                                                   {'name': rfft_name + '/squeeze_real'})
        squeeze_imag = create_op_with_const_inputs(graph, Squeeze, {1: int64_array(-1)},
                                                   {'name': rfft_name + '/squeeze_imag'})

        split_node.out_port(0).connect(squeeze_real.in_port(0))
        split_node.out_port(1).connect(squeeze_imag.in_port(0))
        real_node.out_port(0).get_connection().set_source(squeeze_real.out_port(0))
        imag_node.out_port(0).get_connection().set_source(squeeze_imag.out_port(0))

        rfft_node.out_port(0).connect(split_node.in_port(0))

        rename_nodes([(real_node, real_name + '/to_be_removed'), (squeeze_real, real_name)])
        rename_nodes([(imag_node, imag_name + '/to_be_removed'), (squeeze_imag, imag_name)])

        real_node.in_port(0).disconnect()
        imag_node.in_port(0).disconnect()
