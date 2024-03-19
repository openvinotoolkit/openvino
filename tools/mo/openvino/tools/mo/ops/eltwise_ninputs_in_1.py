# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.error import Error


class EltwiseNin1(Op):
    """
    The elementwise operation that has all inputs in 1 input. This operation is replaced in a front phase with
    a number of simple elementwise operations with 2 inputs.
    """
    op = 'EltwiseNin1'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': None,  # type is None because this operation should not appear in IR
            'infer': None,
            'out_ports_count': 1,
        }, attrs)
        if 'operation' not in self.attrs:
            raise Error('"operation" attribute is not set for operation "{}".'.format(__class__.op))
