# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class DivSqrtDimOp(Op):
    """
    MXNet operation that matches the formula out = (data / sqrt(data.shape[-1])).
    Will be replaced with the corresponding sub-graph
    """
    op = '_contrib_div_sqrt_dim'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)
