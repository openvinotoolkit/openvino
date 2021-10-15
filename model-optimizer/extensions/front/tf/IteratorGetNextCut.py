# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.extractor import add_input_ops
from mo.graph.graph import Graph


class IteratorGetNextCut(FrontReplacementSubgraph):
    """
    Cuts OneShotIterator -> IteratorGetNext pattern
    in order to enable Out Of the Box (OOB) usage.
    Pass is run only if user didn't specify any inputs names and shapes.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf' and graph.graph['cmd_params'].input is None]

    def run_before(self):
        from extensions.front.output_cut import OutputCut
        return [OutputCut]

    def run_after(self):
        from extensions.analysis.json_print import AnalysisJSONPrint
        return [AnalysisJSONPrint]

    def find_and_replace_pattern(self, graph: Graph):
        iter_get_next_shapes = defaultdict(list)
        for iter_get_next in graph.get_op_nodes(op='IteratorGetNext'):
            for port in iter_get_next.out_nodes().keys():
                iter_get_next_shapes[iter_get_next.name].append(dict(
                    shape=iter_get_next.shapes[port],
                    out=port,
                    data_type=iter_get_next.types[port]
                ))

        add_input_ops(graph, iter_get_next_shapes, True)
