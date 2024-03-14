# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.concat import Concat


class CorrectPaddingsForPadAfterComplex(FrontReplacementSubgraph):
    """
    There are TF models with the TF operation Complex that has two real tensors as arguments and returns the complex
    tensor with real and imaginary parts given as arguments in port 0 and 1 respectively.

    Although TF has a native support of complex numbers, OpenVINO doesn't have such support and emulates a complex
    tensor with the shape [N_0, ..., N_{r - 1}] as a real tensor of the shape [N_0, ..., N_{r - 1}, 2] interpreting
    any complex number as a tuple of the form
        (real part, imaginary part)
    That is, the emulated complex tensor has the rank r + 1, not r as in the TF model.

    Hence, when we convert a subgraph of the form

    Complex
       |
       |
      Pad

    we should correct pads_begin and pads_end adding zero at the end of pads_begin and pads_end.

    The transformation performs such corrections.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.tf.pad_tf_to_pad import PadTFToPad
        return [PadTFToPad]

    def pattern(self):
        return dict(
            nodes=[
                ('complex', dict(op='Complex')),
                ('pad', dict(op='Pad')),
            ],
            edges=[
                ('complex', 'pad', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        pad_node = match['pad']
        pads_begin_node = pad_node.in_port(1).get_source().node
        pads_end_node = pad_node.in_port(2).get_source().node

        pads_begin_node_name = pads_begin_node.soft_get('name', pads_begin_node.id)
        pads_end_node_name = pads_end_node.soft_get('name', pads_end_node.id)

        concat_for_pads_begin = create_op_with_const_inputs(graph, Concat,
                                                            {1: int64_array([0])},
                                                            {
                                                                'name': pads_begin_node_name + '/additional',
                                                                'in_ports_count': 2,
                                                                'axis': 0,
                                                            })
        concat_for_pads_end = create_op_with_const_inputs(graph, Concat,
                                                          {1: int64_array([0])},
                                                          {
                                                              'name': pads_end_node_name + '/additional',
                                                              'in_ports_count': 2,
                                                              'axis': 0,
                                                          })
        pad_node.in_port(1).get_connection().insert_node(concat_for_pads_begin)
        pad_node.in_port(2).get_connection().insert_node(concat_for_pads_end)
