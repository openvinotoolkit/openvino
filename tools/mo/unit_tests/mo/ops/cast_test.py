# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.middle.passes.convert_data_type import packed_U4, packed_I4
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import valued_const_with_data, regular_op_with_empty_data, result, build_graph, connect

nodes = lambda value, dst_type: {
    **valued_const_with_data('value', np.array(value)),
    **regular_op_with_empty_data('convert', {'dst_type': dst_type, 'infer': Cast.infer}),
    **result(),
}


class TestCastTest():
    """
    Example of checking:
        7 == 0111,           padded to 0111 0000, results in 112
        7 == 0111, 8 == 1000 packed to 0111 1000, results in 120

        -8 == 1000,          padded to 1000 0000, results in 128
    """

    @pytest.mark.parametrize("value, expected, custom_dtype",[
        ([0], [0], packed_U4),
        ([1], [16], packed_U4),
        ([2], [32], packed_U4),
        ([3], [48], packed_U4),
        ([4], [64], packed_U4),
        ([5], [80], packed_U4),
        ([6], [96], packed_U4),
        ([7], [112], packed_U4),
        ([8], [128], packed_U4),
        ([9], [144], packed_U4),
        ([10], [160], packed_U4),
        ([11], [176], packed_U4),
        ([12], [192], packed_U4),
        ([13], [208], packed_U4),
        ([14], [224], packed_U4),
        ([15], [240], packed_U4),

        ([0, 15], [15], packed_U4),
        ([1, 14], [30], packed_U4),
        ([2, 13], [45], packed_U4),
        ([3, 12], [60], packed_U4),
        ([4, 11], [75], packed_U4),
        ([5, 10], [90], packed_U4),
        ([6, 9], [105], packed_U4),
        ([7, 8], [120], packed_U4),
        ([8, 7], [135], packed_U4),
        ([9, 6], [150], packed_U4),
        ([10, 5], [165], packed_U4),
        ([11, 4], [180], packed_U4),
        ([12, 3], [195], packed_U4),
        ([13, 2], [210], packed_U4),
        ([14, 1], [225], packed_U4),
        ([15, 0], [240], packed_U4),

        ([-8], [128], packed_I4),
        ([-7], [144], packed_I4),
        ([-6], [160], packed_I4),
        ([-5], [176], packed_I4),
        ([-4], [192], packed_I4),
        ([-3], [208], packed_I4),
        ([-2], [224], packed_I4),
        ([-1], [240], packed_I4),
        ([0], [0], packed_I4),
        ([1], [16], packed_I4),
        ([2], [32], packed_I4),
        ([3], [48], packed_I4),
        ([4], [64], packed_I4),
        ([5], [80], packed_I4),
        ([6], [96], packed_I4),
        ([7], [112], packed_I4),

        ([-8, 7], [135], packed_I4),
        ([-7, 6], [150], packed_I4),
        ([-6, 5], [165], packed_I4),
        ([-5, 4], [180], packed_I4),
        ([-4, 3], [195], packed_I4),
        ([-3, 2], [210], packed_I4),
        ([-2, 1], [225], packed_I4),
        ([-1, 0], [240], packed_I4),
        ([0, -1], [15], packed_I4),
        ([1, -2], [30], packed_I4),
        ([2, -3], [45], packed_I4),
        ([3, -4], [60], packed_I4),
        ([4, -5], [75], packed_I4),
        ([5, -6], [90], packed_I4),
        ([6, -7], [105], packed_I4),
        ([7, -8], [120], packed_I4),
    ])
    def test_custom_value_propagation(self, value, expected, custom_dtype):
        graph = build_graph(nodes(value, custom_dtype), [
            *connect('value', 'convert'), *connect('convert', 'output'),
        ])
        partial_infer(graph)

        graph_ref = build_graph(nodes(value, custom_dtype), [
            *connect('value', 'convert'), *connect('convert', 'output')],
                                {'convert_d': {'force_type': custom_dtype, 'force_shape': np.array(value).shape,
                                               'value': expected}})

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp
