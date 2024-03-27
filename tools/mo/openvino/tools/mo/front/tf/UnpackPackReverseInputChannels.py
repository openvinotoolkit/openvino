# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.back.ReverseInputChannels import ReverseChannels
from openvino.tools.mo.front.Pack import Pack
from openvino.tools.mo.front.split_normalizer import AttributedSplitToSplit, SqueezeAxis
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class UnpackPackReverseInputChannels(FrontReplacementSubgraph):
    r"""
    Unpack - Pack nodes sequence from TensorFlow connected like it shown below is a way to ReverseChannels

           /  0 - 2  \
    Unpack -  1 - 1  - Pack
           \  2 - 0  /

    Converting it to internal ReverseChannels node to be fused to Convolution while running ApplyReverseChannels on back
    """
    enabled = True

    def run_before(self):
        # ordering transformations to keep matching pattern as small as possible

        # Unpack from TensorFlow is extracted as AttributedSplit with squeeze_axis=True attribute,
        # so we should execute current transformation before AttributedSplitToSplit and SqueezeAxis

        # Pack from TensorFlow is an operation that creates new dimension, which we add by inserting Unsqueeze on all
        # inputs at Pack transform, so we should execute current transformation before it
        return [AttributedSplitToSplit, Pack, SqueezeAxis]

    def pattern(self):
        return dict(
            nodes=[
                ('unpack', dict(op='AttributedSplit')),
                ('pack', dict(op='Pack')),
            ],
            edges=[
                ('unpack', 'pack', {'out': 0, 'in': 2}),
                ('unpack', 'pack', {'out': 1, 'in': 1}),
                ('unpack', 'pack', {'out': 2, 'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        unpack = match['unpack']
        pack = match['pack']

        if unpack.soft_get('axis', None) is None or unpack.axis != pack.soft_get('axis', None):
            # axes doesn't match - not ReverseChannels case
            return

        axis = unpack.axis

        connected_unpack_ports_count = len([port for port in unpack.out_ports().values() if not port.disconnected()])
        connected_pack_ports_count = len([port for port in pack.in_ports().values() if not port.disconnected()])
        if connected_pack_ports_count != connected_unpack_ports_count or connected_unpack_ports_count != 3:
            # number of connected input ports of Concat and output ports of Split mismatch - not ReverseChannels case
            return

        name = pack.soft_get('name', pack.id)
        log.debug('Unpack - Pack sequence was detected `{}`'.format(name))

        reverse_channels = ReverseChannels(graph, {
            'name': pack.soft_get('name', pack.id) + '/ReverseChannels',
            'axis': int64_array(axis), 'order': int64_array([2, 1, 0])}).create_node()

        pack.out_port(0).get_connection().set_source(reverse_channels.out_port(0))
        unpack.in_port(0).get_connection().set_destination(reverse_channels.in_port(0))
        log.debug('Unpack - Pack was converted to ReverseChannels {}'.format(name))
