# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class AxpyOp(Op):
    """
    Empty Op for Axpy layer. It will be replaced by AxpyToSSandAdd FrontReplacer
    """
    op = 'Axpy'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'infer': None
        }, attrs)
