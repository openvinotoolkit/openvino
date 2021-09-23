# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class AvgPool(BackReplacementPattern):
    """
    Rename Pooling/avg to AvgPool
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('pooling', {'type': 'Pooling', 'pool_method': 'avg'})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pooling']
        node.type = 'AvgPool'
        del node['pool_method']
