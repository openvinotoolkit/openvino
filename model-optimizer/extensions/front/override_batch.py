# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.infer import override_batch


class OverrideBatch(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True

    def find_and_replace_pattern(self, graph: Graph):
        override_batch(graph, graph.graph['cmd_params'].batch)
