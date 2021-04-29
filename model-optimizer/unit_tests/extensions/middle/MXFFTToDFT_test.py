# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from extensions.middle.MXFFTToDFT import MXFFTToDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    regular_op_with_empty_data


fft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 40, 56, 4], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('fft', [8, 40, 56, 4], {'op': 'MXFFT', 'is_inverse': False}),
    **regular_op_with_shaped_data('abs', [8, 40, 56, 4], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

fft_graph_edges = [
    *connect('placeholder', 'fft'),
    *connect('fft', 'abs'),
    *connect('abs', 'output'),
]


converted_fft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 40, 56, 4], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('unsqueeze',
                                 {'type': 'Unsqueeze', 'op': 'Unsqueeze', 'need_shape_inference': True}),
    **valued_const_with_data('unsqueeze_axis', int64_array([-1])),
    **regular_op_with_empty_data('pad',
                                 {'type': 'Pad', 'op': 'Pad', 'need_shape_inference': True}),
    **valued_const_with_data('pads_begin', int64_array([0, 0, 0, 0, 0])),
    **valued_const_with_data('pads_end', int64_array([0, 0, 0, 0, 1])),
    **valued_const_with_data('dft_axes', int64_array([-1])),
    **regular_op_with_empty_data('dft', {'op': 'DFT', 'need_shape_inference': True}),
    **regular_op_with_shaped_data('reshape', [8, 40, 56, 4],
                                  {'type': 'Reshape', 'op': 'Reshape', 'need_shape_inference': True}),
    **valued_const_with_data('new_shape', int64_array([0, 0, 0, -1])),
    **regular_op_with_shaped_data('abs', [8, 40, 56, 4], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

converted_fft_graph_edges = [
    *connect('placeholder', '0:unsqueeze'),
    *connect('unsqueeze_axis', '1:unsqueeze'),
    *connect('unsqueeze', '0:pad'),
    *connect('pads_begin', '1:pad'),
    *connect('pads_end', '2:pad'),
    *connect('pad', '0:dft'),
    *connect('dft_axes', '1:dft'),
    *connect('dft', '0:reshape'),
    *connect('new_shape', '1:reshape'),
    *connect('reshape', 'abs'),
    *connect('abs', 'output'),
]


ifft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 40, 56, 4], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('fft', [8, 40, 56, 4], {'op': 'MXFFT', 'is_inverse': True}),
    **regular_op_with_shaped_data('abs', [8, 40, 56, 4], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

ifft_graph_edges = [
    *connect('placeholder', 'fft'),
    *connect('fft', 'abs'),
    *connect('abs', 'output'),
]


converted_ifft_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', [8, 40, 56, 4], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_empty_data('unsqueeze',
                                 {'type': 'Unsqueeze', 'op': 'Unsqueeze', 'need_shape_inference': True}),
    **valued_const_with_data('unsqueeze_axis', int64_array([-1])),
    **regular_op_with_empty_data('pad',
                                 {'type': 'Pad', 'op': 'Pad', 'need_shape_inference': True}),
    **valued_const_with_data('pads_begin', int64_array([0, 0, 0, 0, 0])),
    **valued_const_with_data('pads_end', int64_array([0, 0, 0, 0, 1])),
    **valued_const_with_data('dft_axes', int64_array([-1])),
    **regular_op_with_empty_data('dft', {'op': 'IDFT', 'need_shape_inference': True}),
    **regular_op_with_empty_data('ss',
                                 {
                                     'op': 'StridedSlice', 'type': 'StridedSlice',
                                     'need_shape_inference': True,
                                     'begin_mask': int64_array([0, 0, 0, 0, 1]),
                                     'end_mask': int64_array([0, 0, 0, 0, 1]),
                                     'ellipsis_mask': int64_array([0, 0, 0, 0, 0]),
                                     'new_axis_mask': int64_array([0, 0, 0, 0, 0]),
                                     'shrink_axis_mask': int64_array([0, 0, 0, 0, 0]),
                                 }),
    **valued_const_with_data('ss_begin', int64_array([0, 0, 0, 0, 0])),
    **valued_const_with_data('ss_end', int64_array([0, 0, 0, 0, 1])),
    **valued_const_with_data('ss_strides', int64_array([1, 1, 1, 1, 1])),
    **regular_op_with_shaped_data('squeeze', [8, 40, 56, 4],
                                  {'type': 'Squeeze', 'op': 'Squeeze', 'need_shape_inference': True}),
    **valued_const_with_data('squeeze_axis', int64_array([-1])),
    **regular_op_with_shaped_data('abs', [8, 40, 56, 4], {'type': 'Abs', 'op': 'Abs'}),
    **result(),
}

converted_ifft_graph_edges = [
    *connect('placeholder', '0:unsqueeze'),
    *connect('unsqueeze_axis', '1:unsqueeze'),
    *connect('unsqueeze', '0:pad'),
    *connect('pads_begin', '1:pad'),
    *connect('pads_end', '2:pad'),
    *connect('pad', '0:dft'),
    *connect('dft_axes', '1:dft'),
    *connect('dft', '0:ss'),
    *connect('ss_begin', '1:ss'),
    *connect('ss_end', '2:ss'),
    *connect('ss_strides', '3:ss'),
    *connect('ss', '0:squeeze'),
    *connect('squeeze_axis', '1:squeeze'),
    *connect('squeeze', 'abs'),
    *connect('abs', 'output'),
]


class MXFFTToDFTTest(unittest.TestCase):
    def test_fft_replacement(self):
        graph = build_graph(nodes_attrs=fft_graph_node_attrs, edges=fft_graph_edges)
        MXFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=converted_fft_graph_node_attrs, edges=converted_fft_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    def test_ifft_replacement(self):
        graph = build_graph(nodes_attrs=ifft_graph_node_attrs, edges=ifft_graph_edges)
        MXFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=converted_ifft_graph_node_attrs, edges=converted_ifft_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
