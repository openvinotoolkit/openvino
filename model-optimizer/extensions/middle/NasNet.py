"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.extractor import add_attrs_props, update_ie_fields
from mo.graph.graph import Node, Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op


class NasNet(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        return []

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

        pad_op = match['pad_op']

        sslice = match['sslice']
        sslice_out = match['sslice_out']
        begin = []
        end = []
        stride = []
        for s in sslice.slices:
            begin.append(s.start)
            end.append(s.stop)
            stride.append(s.step)

        if not np.array_equal(pad_op.pads, np.array([[0, 0], [0, 1], [0, 1], [0, 0]])):
            log.error(" Pad values doesn't match!")
            return

        if not np.array_equal(begin, np.array([0, 1, 1, 0])):
            log.error("StridedSlice has wrong begin")
            return

        if not np.array_equal(sslice.end_mask, np.array([0, 0, 0, 0])) or not np.array_equal(sslice.begin_mask, np.array([0, 1, 1, 0])):
            log.error("StridedSlice has wrong masks")
            return

        # Cut Smth-x->Pad->StrudedSlice-x->AvgPool
        graph.remove_edge(input.id, pad_op.id)
        graph.remove_edge(sslice.id, sslice_out.id)

        # Pad -> Conv
        conv_node = graph.unique_id(pad_op.name + '/Conv_')
        conv_weights_node = graph.unique_id(pad_op.name + '/ConvW_')
        conv_weights = np.ones((input.shape[3], 1, 1, 1))
        conv_output = graph.unique_id(pad_op.name + '/ConvOut_')
        output_shape = np.array([input.shape[0], input.shape[1] + 1, input.shape[2] + 1, input.shape[3]])

        graph.add_node(conv_node,
                       **add_attrs_props(
                           dict(kind='op', precision="FP32", type='Convolution', name=conv_node, op='Conv2D',
                                stride=np.array([1, 1, 1, 1]), dilation=np.array([1, 1, 1, 1]),
                                group=input.shape[3], bias_addable=True, bias_term=False,
                                spatial_dims=np.array([1, 2]),
                                kernel_spatial=np.array([1, 1]),
                                pad=np.array([[0, 0], [0, 1], [0, 1], [0, 0]]), output_shape=output_shape,
                                channel_dims=np.array([3]),
                                in_ports_count=3, out_ports_count=1)))

        graph.add_node(conv_weights_node, **add_attrs_props(
            dict(kind='data', precision="FP32", name=conv_weights_node, value=np.array(conv_weights),
                 shape=np.array(conv_weights.shape),
                 data_type=input.data_type, infer=None,
                 spatial_dims=np.array([0, 1]),
                 input_channel_dim=2,
                 output_channel_dim=3,
                 dims_number=4, can_be_bias=True)))
        graph.add_node(conv_output, **add_attrs_props(
            dict(kind='data', precision="FP32", name=conv_output, value=None, shape=output_shape,
                 data_type=input.data_type)))

        # StridedSlice -> Crop
        crop_cls = Op.get_op_class_by_name('Crop')
        crop = crop_cls(graph, dict(name=sslice.name + '/Crop_', axis=np.array([1, 2]),
                                    dim=np.array([output_shape[1] - 1, output_shape[2] - 1]), offset=np.array([1, 1])))
        crop.create_node_with_data([Node(graph, conv_output)], data_nodes=sslice_out)

        # Connect : Conv->Crop->AvgPool
        graph.add_edges_from([
            (input.id, conv_node, {'in': 0}),
            (conv_weights_node, conv_node, {'in': 1, 'bin': 'weights'}),
            (conv_node, conv_output, {'out': 0}),
        ])
        update_ie_fields(graph.node[conv_node], graph.graph['ir_version'])
