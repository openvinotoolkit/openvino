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

import numpy as np

from extensions.middle.pass_separator import PostMiddleStart
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.convolution import Convolution
from mo.ops.crop import Crop


class NasNet(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        return [PostMiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('pad_op', dict(kind='op', op='Pad')),
                ('pad_out', dict(kind='data')),

                ('begin', dict(kind='data')),
                ('end', dict(kind='data')),
                ('stride', dict(kind='data')),

                ('sslice', dict(kind='op', op='StridedSlice')),
                ('sslice_out', dict(kind='data')),

                ('avg_pool', dict(kind='op', op='AvgPool')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'pad_op', {'in': 0}),
                ('pad_op', 'pad_out'),

                ('begin', 'sslice', {'in': 1}),
                ('end', 'sslice', {'in': 2}),
                ('stride', 'sslice', {'in': 3}),

                ('pad_out', 'sslice', {'in': 0}),
                ('sslice', 'sslice_out'),

                ('sslice_out', 'avg_pool', {'in': 0}),
                ('avg_pool', 'output')
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        """
        Converts specific for NasNet topology subgraph Pad->StridedSlice->AvgPool to Conv->Crop->AvgPool
        """
        input = match['input']

        pad_node = match['pad_op']
        pad_node_name = pad_node.soft_get('name', pad_node.id)

        sslice_node = match['sslice']
        begin = []
        end = []
        stride = []
        for s in sslice_node.slices:
            begin.append(s.start)
            end.append(s.stop)
            stride.append(s.step)

        pads_begin = pad_node.in_port(1).data.get_value()
        pads_end = pad_node.in_port(2).data.get_value()
        if pads_begin is None or pads_end is None:
            log.error('Pad values for node "{}" are not constants'.format(pad_node_name))
            return

        if not np.array_equal(pads_begin, int64_array([0, 0, 0, 0])):
            log.error('Pad begin values doesn\'t match for node {}!'.format(pad_node_name))
            return

        if not np.array_equal(pads_end, int64_array([0, 1, 1, 0])):
            log.error('Pad end values doesn\'t match for node {}!'.format(pad_node_name))
            return

        if not np.array_equal(begin, int64_array([0, 1, 1, 0])):
            log.error("StridedSlice has wrong begin")
            return

        if not np.array_equal(sslice_node.end_mask, int64_array([0, 0, 0, 0])) or not np.array_equal(sslice_node.begin_mask,
                                                                                                int64_array(
                                                                                                    [0, 1, 1, 0])):
            log.error("StridedSlice has wrong masks")
            return

        # Pad -> Conv
        conv_name = graph.unique_id(pad_node.name + '/Conv_')
        conv_weights_name = graph.unique_id(pad_node.name + '/ConvW_')
        conv_weights = np.ones((input.shape[3], 1, 1, 1))
        output_shape = int64_array([input.shape[0], input.shape[1] + 1, input.shape[2] + 1, input.shape[3]])

        conv_node = Convolution(graph, dict(name=conv_name,
                                            stride=int64_array([1, 1, 1, 1]),
                                            dilation=int64_array([1, 1, 1, 1]),
                                            group=input.shape[3],
                                            bias_addable=True,
                                            bias_term=False,
                                            spatial_dims=int64_array([1, 2]),
                                            kernel_spatial=int64_array([1, 1]),
                                            pad=int64_array([[0, 0], [0, 1], [0, 1], [0, 0]]),
                                            output_shape=output_shape,
                                            batch_dims=int64_array([0]),
                                            channel_dims=int64_array([3]),
                                            output=input.shape[3],
                                            input_feature_channel=1,
                                            output_feature_channel=0,
                                            )).create_node()

        weights_const_node = Const(graph, dict(name=conv_weights_name, value=conv_weights,
                                          shape=int64_array(conv_weights.shape))).create_node()

        # StridedSlice -> Crop
        crop_node = Crop(graph, dict(name=sslice_node.name + '/Crop_', axis=int64_array([1, 2]),
                                dim=int64_array([output_shape[1] - 1, output_shape[2] - 1]), offset=int64_array([1, 1]))
                    ).create_node()

        # Connect nodes
        pad_node.in_port(0).get_connection().set_destination(conv_node.in_port(0))
        weights_const_node.out_port(0).connect(conv_node.in_port(1))
        conv_node.out_port(0).connect(crop_node.in_port(0))
        sslice_node.out_port(0).get_connection().set_source(crop_node.out_port(0))

        conv_node.in_port(1).bin = 'weights'

        # Remove Pad and StridedSlice nodes from graph
        graph.remove_node(pad_node.id)
        graph.remove_node(sslice_node.id)
