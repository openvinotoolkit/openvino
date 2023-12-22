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
        7 == 0111,           padded to 00000111, results in 7
        7 == 0111, 8 == 1000 packed to 10000111, results in 7+16

        -8 == 1000,          padded to 00001000, results in 8
    """

    @pytest.mark.parametrize("value, expected, custom_dtype",
        [([i], [i], packed_U4) for i in range(16)] +
        [([i, 15-i], [i + (15-i)*16], packed_U4) for i in range(16)] +
        [([-i], [16-i], packed_I4) for i in range(1, 8+1)] +
        [([i], [i], packed_I4) for i in range(8)] +
        [([-i-1, i], [16-i-1 + 16*i], packed_I4) for i in range(8)] +
        [([i, -i-1], [i + 16*(16-i-1)], packed_I4) for i in range(8)]
    )
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
