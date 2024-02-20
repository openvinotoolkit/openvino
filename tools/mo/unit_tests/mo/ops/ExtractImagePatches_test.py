# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pytest

import numpy as np

from openvino.tools.mo.ops.ExtractImagePatches import ExtractImagePatches
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes = {
    'input': {'op': 'Parameter', 'kind': 'op', 'shape': None},
    'input_data': {'value': None, 'kind': 'data', 'shape': None},
    'EIP': {'op': 'ExtractImagePatches', 'kind': 'op', 'sizes': None, 'strides': None, 'rates': None, 'auto_pad': None},
    'EIP_data': {'value': None, 'kind': 'data', 'shape': None},
    'output': {'op': 'Result', 'kind': 'op', 'shape': None},
}

edges = [
    ('input', 'input_data'),
    ('input_data', 'EIP'),
    ('EIP', 'EIP_data'),
    ('EIP_data', 'output'),
]

class TestExtractImagePatchesPartialInfer():
    @pytest.mark.parametrize("input_shape, sizes, strides, rates, auto_pad, layout, output_shape",[
        ([1, 10, 10, 3], [1, 3, 3, 1], [1, 5, 5, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 2, 2, 27]),
        ([1, 10, 10, 3], [1, 3, 3, 1], [1, 5, 5, 1], [1, 2, 2, 1], 'valid', 'NHWC', [1, 2, 2, 27]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 8, 8, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 1, 1, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 8, 8, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 2, 2, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 2, 2, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'same_lower', 'NHWC', [1, 2, 2, 48]),
        ([1, 64, 64, 3], [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 62, 62, 27]),
        ([1, 64, 64, 3], [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 64, 64, 27]),

        ([1, 3, 10, 10], [1, 1, 3, 3], [1, 1, 5, 5], [1, 1, 1, 1], 'valid', 'NCHW', [1, 27, 2, 2]),
        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 8, 8], [1, 1, 1, 1], 'valid', 'NCHW', [1, 48, 1, 1]),

        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 9, 9], [1, 1, 1, 1], 'same_upper', 'NCHW', [1, 48, 2, 2]),
        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 9, 9], [1, 1, 1, 1], 'same_lower', 'NCHW', [1, 48, 2, 2]),

    ])


    def test_eip_infer(self, input_shape, sizes, strides, rates, auto_pad, layout, output_shape):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=edges,
            update_attributes={
                'input': {'shape': int64_array(input_shape)},
                'input_data': {'shape': int64_array(input_shape)},
                'EIP': {'spatial_dims': int64_array([1, 2]) if layout == 'NHWC' else int64_array([2, 3]),
                        'sizes': int64_array(sizes), 'strides': int64_array(strides), 'rates': int64_array(rates),
                        'auto_pad': auto_pad},
            }
        )

        graph.graph['layout'] = layout

        eip_node = Node(graph, 'EIP')
        ExtractImagePatches.infer(eip_node)

        assert np.array_equal(eip_node.out_port(0).data.get_shape(), output_shape)
