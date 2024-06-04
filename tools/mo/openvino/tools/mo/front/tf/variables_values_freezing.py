# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.loader import variables_to_constants
from openvino.tools.mo.graph.graph import Graph


class VariablesToConstants(FrontReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['variables_values']]

    def run_after(self):
        from openvino.tools.mo.front.input_cut import InputCut
        return [InputCut]

    def run_before(self):
        from openvino.tools.mo.front.freeze_placeholder_value import FreezePlaceholderValue
        return [FreezePlaceholderValue]

    def find_and_replace_pattern(self, graph: Graph):
        variables_to_constants(graph, graph.graph['variables_values'])
        del graph.graph['variables_values']
