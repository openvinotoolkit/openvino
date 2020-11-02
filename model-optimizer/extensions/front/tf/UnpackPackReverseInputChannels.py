"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log

from extensions.back.ReverseInputChannels import ReverseChannels
from extensions.front.Pack import Pack
from extensions.front.split_normalizer import AttributedSplitToSplit, SqueezeAxis
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class UnpackPackReverseInputChannels(FrontReplacementSubgraph):
    """
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
            'axis': int64_array(axis), 'order': int64_array([2, 0, 1])}).create_node()

        pack.out_port(0).get_connection().set_source(reverse_channels.out_port(0))
        unpack.in_port(0).get_connection().set_destination(reverse_channels.in_port(0))
        log.debug('Unpack - Pack was converted to ReverseChannels {}'.format(name))
