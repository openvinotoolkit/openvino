# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.extractor import add_input_ops
from openvino.tools.mo.graph.graph import Graph


class InputCut(FrontReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_after(self):
        from openvino.tools.mo.front.output_cut import OutputCut
        return [OutputCut]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        add_input_ops(graph, graph.graph['user_shapes'], True)
