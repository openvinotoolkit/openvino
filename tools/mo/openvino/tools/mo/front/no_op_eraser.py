# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class NoOpEraser(FrontReplacementSubgraph):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('noop', dict(kind='op', op='NoOp')),
                   ('output', dict(kind='op', op='Result'))
                   ],
            edges=[('noop', 'output')]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        graph.erase_node(match['output'])
        graph.erase_node(match['noop'])
        log.info("NoOp node \"{}\" was removed from the graph".format(match['noop'].id))
