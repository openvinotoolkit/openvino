# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class StandaloneConstEraser(FrontReplacementSubgraph):
    enabled = True
    # TODO: remove this transformation once all plugins support constant value network.
    # Now it avoids to be run recursively since Const->Result sub-graph can be encountered in a body graph of Loop node
    run_not_recursively = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('const', dict(kind='op', op='Const')),
                   ('output', dict(kind='op', op='Result'))
                   ],
            edges=[('const', 'output')]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        if not len(match['const'].in_edges()) and len(match['const'].out_edges()) == 1:
            graph.erase_node(match['const'])
            graph.erase_node(match['output'])
            log.info("Standalone Const node \"{}\" was removed from the graph".format(match['const'].id))
