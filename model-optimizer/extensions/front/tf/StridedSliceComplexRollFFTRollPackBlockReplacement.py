# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0



import logging as log

import numpy as np

from extensions.front.Pack import Pack
from extensions.front.tf.nearest_neighbor_upsampling import NearestNeighborUpsampling
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from extensions.ops.mvn import MVN
from extensions.ops.dft import DFT, IDFT
from mo.front.tf.graph_utils import create_op_with_const_inputs


class StridedSliceComplexRollFFTRollPackBlockReplacement(FrontReplacementSubgraph):
    enabled = False

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(op='StridedSlice')),
                ('strided_slice_imag', dict(op='StridedSlice')),
                ('complex', dict(op='Complex')),
                ('roll', dict(op='Roll')),
                ('fft', dict(op='TFFFT')),
                ('unroll', dict(op='Roll')),
                ('real', dict(op='Real')),
                ('imag', dict(op='Imag')),
                ('pack', dict(op='Pack')),
            ],
            edges=[
                ('strided_slice_real', 'complex', {'in': 0}),
                ('strided_slice_imag', 'complex', {'in': 1}),
                ('complex', 'roll', {'in': 0}),
                ('roll', 'fft2d', {'in': 0}),
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


def create_dft_from_tffft(graph: Graph, tffft: Node) -> Node:
    num_of_dims = tffft.soft_get('num_of_dimensions', 1)
    axes = int64_array(range(-num_of_dims, 0))
    if tffft.soft_get('is_inverse', False):
        return create_op_with_const_inputs(graph, DFT, {1: axes}, {})
    else:
        return create_op_with_const_inputs(graph, IDFT, {1: axes}, {})