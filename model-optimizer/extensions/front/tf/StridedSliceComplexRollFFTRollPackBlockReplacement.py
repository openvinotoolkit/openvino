# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log

from extensions.ops.dft import DFT, IDFT
from extensions.ops.roll import TFRoll
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes


class StridedSliceComplexRollFFTRollPackBlockReplacement(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(op='StridedSlice')),
                ('strided_slice_imag', dict(op='StridedSlice')),
                ('complex', dict(op='Complex')),
                ('roll', dict(op='TFRoll')),
                ('fft', dict(op='TFFFT')),
                ('unroll', dict(op='TFRoll')),
                ('real', dict(op='Real')),
                ('imag', dict(op='Imag')),
                ('pack', dict(op='Pack')),
            ],
            edges=[
                ('strided_slice_real', 'complex', {'in': 0}),
                ('strided_slice_imag', 'complex', {'in': 1}),
                ('complex', 'roll', {'in': 0}),
                ('roll', 'fft', {'in': 0}),
                ('fft', 'unroll', {'in': 0}),
                ('unroll', 'real', {'in': 0}),
                ('unroll', 'imag', {'in': 0}),
                ('real', 'pack', {'in': 0}),
                ('imag', 'pack', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']

        if strided_slice_real['pb'].input[0] != strided_slice_imag['pb'].input[0]:
            log.debug('The pattern does not correspond to (i)fftxd with shift. Different inputs.')
            return

        roll = match['roll']
        unroll = match['unroll']
        roll_name = roll.soft_get('name', roll.id)
        unroll_name = unroll.soft_get('name', unroll.id)

        roll_before = TFRoll(graph, {'need_axes_correction': True}).create_node()
        roll_after = TFRoll(graph, {'need_axes_correction': True}).create_node()

        roll.in_port(1).get_connection().set_destination(roll_before.in_port(1))
        roll.in_port(2).get_connection().set_destination(roll_before.in_port(2))

        strided_slice_real = match['strided_slice_real']
        strided_slice_real.in_port(0).get_connection().set_destination(roll_before.in_port(0))

        tf_fft = match['fft']
        tf_fft_name = tf_fft.soft_get('name', tf_fft.id)

        unroll.in_port(1).get_connection().set_destination(roll_after.in_port(1))
        unroll.in_port(2).get_connection().set_destination(roll_after.in_port(2))

        dft_node = create_dft_from_tffft(graph, tf_fft, roll_before)
        dft_node.out_port(0).connect(roll_after.in_port(0))

        pack = match['pack']
        pack.out_port(0).get_connection().set_source(roll_after.out_port(0))

        rename_nodes([(roll, roll_name + '/to_be_removed'), (roll_before, roll_name)])
        rename_nodes([(unroll, unroll_name + '/to_be_removed'), (roll_after, unroll_name)])
        rename_nodes([(tf_fft, tf_fft_name + '/to_be_removed'), (dft_node, tf_fft_name)])

        if not graph.graph['cmd_params'].disable_nhwc_to_nchw or graph.graph['layout'] == 'NHWC':
            dft_node['need_insert_transposes_for_dft'] = True


def create_dft_from_tffft(graph: Graph, tffft: Node, input_node=None) -> Node:
    num_of_dims = tffft.soft_get('num_of_dimensions', 1)
    axes = int64_array(range(-num_of_dims, 0))
    op = IDFT if tffft.soft_get('is_inverse', False) else DFT
    return create_op_with_const_inputs(graph, op, {1: axes}, {'in_ports_count': 2}, input_node)
