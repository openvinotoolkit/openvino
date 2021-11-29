# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.back.LayoutChangeForEinsum import LayoutChangeForEinsum
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect

nodes_attributes = {
    # Parameter layers
    **regular_op_with_shaped_data('placeholder_1', None, {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_2', None, {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_3', None, {'type': 'Parameter', 'op': 'Parameter'}),

    # Einsum layer
    **regular_op_with_shaped_data('einsum', None, {'type': 'Einsum', 'op': 'Einsum'}),

    # Result layer
    **result(),

    # Transpose layers
    **regular_op_with_shaped_data('transpose_1', None,
                                  {'type': 'Transpose', 'op': 'Transpose', 'need_shape_inference': True}),
    **regular_op_with_shaped_data('transpose_3', None,
                                  {'type': 'Transpose', 'op': 'Transpose', 'need_shape_inference': True}),

    # Const layers
    **valued_const_with_data('axis_1_const', int64_array([0, 2, 3, 1])),
    **valued_const_with_data('axis_3_const', int64_array([0, 4, 1, 2, 3])),
}


class LayoutChangeForEinsumTests(unittest.TestCase):
    def test_layout_change_einsum(self):
        graph = build_graph(nodes_attributes,
                            [*connect('placeholder_1', '0:einsum'),
                             *connect('placeholder_2', '1:einsum'),
                             *connect('placeholder_3', '2:einsum'),
                             *connect('einsum', 'output')],
                            {  # this input stays as is since it is of a rank equal to 3
                                'placeholder_1_d': {'shape': np.array([2, 3, 5])},
                                # [3, 5, 7, 8] - NHWC, [3, 8, 5, 7] - NCHW
                                # this input does not require additional transpose
                                # since the corresponding subscript can be adjusted
                                'placeholder_2_d': {'shape': np.array([3, 8, 5, 7])},
                                # [3, 5, 10, 12] - NHWC, [3, 12, 5, 10] - NCHW
                                # the third input must be transposed to NHWC layout
                                # since ellipsis covers multiple dimensions in the end
                                # the corresponding subscript is not changed
                                'placeholder_3_d': {'shape': np.array([3, 12, 8, 10])},
                                # equation is still for NHWC layout
                                'einsum': {'equation': "abc,bcde,bc...->ade..."},
                                # [2, 7, 8, 10, 12] - NHWC, [2, 12, 7, 8, 10] - NCHW
                                # the output is in NCHW layout but its shape will be re-inferred since
                                # the output stays in NHWC layout due to ellipsis in the end
                                # and additional transpose to NCHW will be inserted
                                'einsum_d': {'shape': np.array([2, 12, 7, 8, 10])},
                            }, nodes_with_edges_only=True)
        graph.graph['fw'] = 'tf'

        graph_ref = build_graph(nodes_attributes,
                                [*connect('placeholder_3', '0:transpose_1'),
                                 *connect('axis_1_const', '1:transpose_1'),
                                 *connect('placeholder_1', '0:einsum'),
                                 *connect('placeholder_2', '1:einsum'),
                                 *connect('transpose_1', '2:einsum'),
                                 *connect('einsum', '0:transpose_3'),
                                 *connect('axis_3_const', '1:transpose_3'),
                                 *connect('transpose_3', 'output')],
                                {'placeholder_1_d': {'shape': np.array([2, 3, 5])},
                                 'placeholder_2_d': {'shape': np.array([3, 8, 5, 7])},
                                 'einsum': {'equation': "abc,becd,bc...->ade..."},
                                 'einsum_d': {'shape': np.array([2, 12, 7, 8, 10])}
                                 })

        LayoutChangeForEinsum().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
