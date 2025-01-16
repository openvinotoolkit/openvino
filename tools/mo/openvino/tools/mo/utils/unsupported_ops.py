# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections

from openvino.tools.mo.graph.graph import Node, Graph


class UnsupportedOps(object):
    def __init__(self, graph: Graph):
        self.graph = graph
        # map op to a list of node names
        self.unsupported = collections.defaultdict(list)

    def add(self, node: Node):
        op = node.op if node.has_valid('op') else '<UNKNOWN OP>'
        name = node.name if node.has_valid('name') else '<UNKNOWN NAME>'
        self.unsupported[op].append(name)

    def report(self, reporter, header=None):
        if len(self.unsupported) > 0:
            if header:
                reporter(header)
            for k, v in self.unsupported.items():
                reporter(' ' * 4 + str(k) + ' (' + str(len(v)) + ')')
                for node_name in v:
                    reporter(' ' * 8 + node_name)
