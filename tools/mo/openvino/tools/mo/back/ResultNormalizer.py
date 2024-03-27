# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class ResultNormalizer(BackReplacementPattern):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('result', {'type': 'Result'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['result']
        if len(node.in_nodes()) == 0:
            graph.erase_node(node)
