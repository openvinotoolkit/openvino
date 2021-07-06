# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.flatten_to_reshape import FlattenToReshape
from extensions.front.mxnet.ssd_pattern_remove_reshape import SsdPatternRemoveReshape
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class SsdPatternRemoveFlatten(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [SsdPatternRemoveReshape, FlattenToReshape]

    def pattern(self):
        return dict(
            nodes=[
                ('multi_box_prior', dict(op='_contrib_MultiBoxPrior')),
                ('flatten', dict(op='Flatten'))
            ],
            edges=[
                ('multi_box_prior', 'flatten', {'in': 0})
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find each occurrence of pattern: _contrib_MultiBoxPrior -> Flatten
        remove Flatten layer - IE does not expect outputs to be flattened

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        graph.erase_node(match['flatten'])
