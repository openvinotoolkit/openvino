# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import add_input_ops
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class MiddleInputCut(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.ScaleInput import ScaleInput
        return [ScaleInput]

    def find_and_replace_pattern(self, graph: Graph):
        add_input_ops(graph, graph.graph['user_shapes'], False)
