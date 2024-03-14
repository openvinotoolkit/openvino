# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class RemoveIdentity(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.InputCut import MiddleInputCut
        return [MiddleInputCut]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', identity=True))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])


class RemoveDropout(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.InputCut import MiddleInputCut
        return [MiddleInputCut]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(op='Dropout'))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])


class RemoveNodesWithZeroPhase(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.InputCut import MiddleInputCut
        return [MiddleInputCut]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', phase=0))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])
