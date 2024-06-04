# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.infer import override_batch


class OverrideBatch(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        override_batch(graph, graph.graph['cmd_params'].batch)
