# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class MaxPool(BackReplacementPattern):
    """
    Rename Pooling/max to MaxPool
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('pooling', {'type': 'Pooling', 'pool_method': 'max'})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pooling']
        node.type = 'MaxPool'
        del node['pool_method']
        if 'exclude_pad' in node:
            del node['exclude_pad']
