# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class DeleteControlFlowEdges(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.PartialInfer import PartialInfer
        return [PartialInfer]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for u, v, k, attrs in list(graph.edges(keys=True, data=True)):
            if 'control_flow_edge' in attrs and attrs['control_flow_edge']:
                graph.remove_edge(u, v, k)
                log.debug('Removing control flow edge from {} to {}'.format(u, v))
