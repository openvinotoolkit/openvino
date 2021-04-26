# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.back.LayoutChangeForEinsum import LayoutChangeForEinsum
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph
from mo.front.common.partial_infer.utils import int64_array


nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Einsum
    'einsum': {'type': 'Einsum', 'kind': 'op', 'op': 'Einsum'},
    'einsum_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Result layer
    'result': {'type': 'Result', 'kind': 'op', 'op': 'Result'},
    # Transpose layers
    'transpose_1': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axis_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'axis_1_const_data': {'kind': 'data', 'value': None, 'shape': None},
    'transpose_3': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose', 'need_shape_inference': True},
    'transpose_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    'axis_3_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'axis_3_const_data': {'kind': 'data', 'value': None, 'shape': None},
}


class LayoutChangeForEinsumTests(unittest.TestCase):
    def test_layout_change_einsum(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_3', 'placeholder_3_data'),
                             ('placeholder_1_data', 'einsum'),
                             ('placeholder_2_data', 'einsum'),
                             ('placeholder_3_data', 'einsum'),
                             ('einsum', 'einsum_data'),
                             ('einsum_data', 'result'),
                             ],
                            {# this input stays as is since it is of a rank equal to 3
                             'placeholder_1_data': {'shape': np.array([2, 3, 5])},
                             # [3, 5, 7, 8] - NHWC, [3, 8, 5, 7] - NCHW
                             # this input does not require additional transpose
                             # since the corresponding subscript can be adjusted
                             'placeholder_2_data': {'shape': np.array([3, 8, 5, 7])},
                             # [3, 5, 10, 12] - NHWC, [3, 12, 5, 10] - NCHW
                             # the third input must be transposed to NHWC layout
                             # since ellipsis covers multiple dimensions in the end
                             # the corresponding subscript is not changed
                             'placeholder_3_data': {'shape': np.array([3, 12, 8, 10])},
                             # equation is still for NHWC layout
                             'einsum': {'equation': "abc,bcde,bc...->ade..."},
                             # [2, 7, 8, 10, 12] - NHWC, [2, 12, 7, 8, 10] - NCHW
                             # the output is in NCHW layout but its shape will be re-inferred since
                             # the output stays in NHWC layout due to ellipsis in the end
                             # and additional transpose to NCHW will be inserted
                             'einsum_data': {'shape': np.array([2, 12, 7, 8, 10])},
                             }, nodes_with_edges_only=True)
        graph.graph['fw'] = 'tf'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_3', 'placeholder_3_data'),
                                 ('placeholder_3_data', 'transpose_1'),
                                 ('axis_1_const', 'axis_1_const_data'),
                                 ('axis_1_const_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),

                                 ('placeholder_1_data', 'einsum'),
                                 ('placeholder_2_data', 'einsum'),
                                 ('transpose_1_data', 'einsum'),

                                 ('einsum', 'einsum_data'),
                                 ('einsum_data', 'transpose_3'),
                                 ('axis_3_const', 'axis_3_const_data'),
                                 ('axis_3_const_data', 'transpose_3'),
                                 ('transpose_3', 'transpose_3_data'),
                                 ('transpose_3_data', 'result'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([2, 3, 5])},
                                 'placeholder_2_data': {'shape': np.array([3, 8, 5, 7])},
                                 'axis_1_const_data': {'value': int64_array([0, 2, 3, 1])},
                                 'einsum': {'equation': "abc,becd,bc...->ade..."},
                                 'einsum_data': {'shape': np.array([2, 12, 7, 8, 10])},
                                 'axis_3_const_data': {'value': int64_array([0, 4, 1, 2, 3])},
                                 })

        LayoutChangeForEinsum().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
