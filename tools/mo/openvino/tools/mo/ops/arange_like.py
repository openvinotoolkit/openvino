# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class ArangeLikeOp(Op):
    """
    MXNet operation which returns a sequence of numbers. If axis attribute is None, the output has the
    same shape as the input. Otherwise, the output is a 1D array with size of the specified axis.

    Attributes:
        start - Start of interval
        step - Spacing between values
        repeat - The repeating time of all elements. Now we can support only default value (= 1)
        axis - Arange elements according to the size of a certain axis of input array. Defualt value is None

    """
    op = 'arange_like'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)