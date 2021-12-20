# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class MXNetBatchDot(Op):
    """
    MXNet operation which computes matrix multiplication of x and y similar to TF or ONNX MatMul operation.

    Attributes:
        transpose_a - if true then transpose the first input before multiplication
        transpose_b - if true then transpose the second input before multiplication
    """
    op = 'batch_dot'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'transpose_a': False,
            'transpose_b': False
        }
        super().__init__(graph, mandatory_props, attrs)
