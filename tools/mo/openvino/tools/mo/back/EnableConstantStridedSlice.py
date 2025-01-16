# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class EnableConstantStridedSlice(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].static_shape]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('const_strided_slice', {'op': 'StridedSlice', 'type': lambda type: type != 'StridedSlice'}),
                   ('data', {'kind': 'data', 'value': lambda value: value is not None})
                   ],
            edges=[('const_strided_slice', 'data')],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        graph.node[match['const_strided_slice'].id]['type'] = 'StridedSlice'
