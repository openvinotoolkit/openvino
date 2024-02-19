# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class Concat(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('concat', dict(op='Concat', simple_concat=True))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        There are Concat and ConcatV2 operations in TensorFlow
        The main difference is incoming port of tensor representing axis of concatenation
        In Concat it is the 0 port, in ConcatV2 it is the last port
        To reuse ConcatV2 logic (infer) that already exists in the Model Optimizer here we renumber ports of Concat
        """
        in_edges = list(graph.in_edges(match['concat'].id, data=True))
        for u, v, attrs in in_edges:
            in_port = attrs['in']
            attrs['in'] = len(in_edges) - 1 if in_port == 0 else attrs['in'] - 1
        if match['concat'].has('axis'):
            # we delete axis parameter here (it was set by default by Concat Op) to carefully get it from the last
            # input in Concat infer function
            del graph.node[match['concat'].id]['axis']
