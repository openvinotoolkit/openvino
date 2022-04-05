# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class InterpolateTranspose(FrontReplacementSubgraph):
    """
    Delete useless transposes around ResizeNearestNeighbor op. In TF this op is working in NHWC layout,
    Resample in OpenVINO working in NCHW layout. If all graph has NCHW layout we should delete transposes around
    Resample: (NCHW->NHWC) -> Resample -> (NHWC -> NCHW) to run this op in NCHW without changes of layout.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    def pattern(self):
        return dict(
            nodes=[
                ('interpolate',
                 {
                     'kind': 'op',
                     'op': 'Interpolate'
                 }),
                ('transpose_1', {'kind': 'op', 'op': 'Transpose'}),
                ('transpose_1_order',
                 {
                     'kind': 'op',
                     'op': 'Const',
                     'value': lambda value: value is not None and np.array_equal(value, int64_array([0, 2, 3, 1]))
                 }),
                ('transpose_2', {'kind': 'op', 'op': 'Transpose'}),
                ('transpose_2_order',
                 {
                     'kind': 'op',
                     'op': 'Const',
                     'value': lambda value: value is not None and np.array_equal(value, int64_array([0, 3, 1, 2]))
                 }),
            ],
            edges=[
                ('transpose_1', 'interpolate', {'in': 0, 'out': 0}),
                ('transpose_1_order', 'transpose_1', {'in': 1, 'out': 0}),
                ('interpolate', 'transpose_2', {'in': 0, 'out': 0}),
                ('transpose_2_order', 'transpose_2', {'in': 1, 'out': 0}),
            ]
        )

    def run_after(self):
        from openvino.tools.mo.front.InterpolateNormalizer import InterpolateNormalizer
        return [InterpolateNormalizer]

    def replace_sub_graph(self, graph: Graph, match: dict):
        interpolate = match['interpolate']
        transpose_1 = match['transpose_1']
        transpose_2 = match['transpose_2']

        axes = Interpolate.get_axes(interpolate)
        if axes is None or not np.array_equal(axes, int64_array([1, 2])):
            return

        # because we remove Transpose layers the ResizeNearestNeighbor should be updated for NCHW layout
        opset = interpolate.get_opset()
        assert opset in ['opset1', 'opset4'], \
            'Interpolate node with name {} has unsupported opset'.format(interpolate.soft_get('name', interpolate.id))
        if opset == 'opset1':
            interpolate.axes = int64_array([2, 3])
        else:
            interpolate.in_port(3).data.set_value(int64_array([2, 3]))

        transpose_1.in_port(0).get_connection().set_destination(interpolate.in_port(0))
        transpose_2.out_port(0).get_connection().set_source(interpolate.out_port(0))

        graph.remove_nodes_from([transpose_1.id, transpose_2.id])
