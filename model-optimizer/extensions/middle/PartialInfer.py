# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.middle.passes.infer import partial_infer
from mo.middle.replacement import MiddleReplacementPattern


class PartialInfer(MiddleReplacementPattern):
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        partial_infer(graph)
