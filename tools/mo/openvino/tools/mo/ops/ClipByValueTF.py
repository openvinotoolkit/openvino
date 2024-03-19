# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class ClibByValueTF(Op):
    """
    The ClipByValue from TF which will be replaced with a front transformation.
    """
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': 'ClipByValueTF',
            'out_ports_count': 1,
            'in_ports_count': 3,
            'infer': None
        }
        super().__init__(graph, mandatory_props, attrs)
