# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementPattern
from mo.front.extractor import add_input_ops
from mo.graph.graph import Graph


class InputCut(FrontReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.output_cut import OutputCut
        return [OutputCut]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        add_input_ops(graph, graph.graph['user_shapes'], True)
