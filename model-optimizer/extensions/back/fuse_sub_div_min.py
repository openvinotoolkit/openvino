# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Sub, Div, Negative
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph


class Negate(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('mul', {'type': 'Multiply'})
            ],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        mul = match['mul']
        name = mul.soft_get('name', mul.id)

        mul_POS_port = None
        if mul.in_port(0).data.get_value() is not None and np.all(mul.in_port(0).data.get_value() == -1):
            mul_POS_port = mul.in_port(1)
        if mul.in_port(1).data.get_value() is not None and np.all(mul.in_port(1).data.get_value() == -1):
            mul_POS_port = mul.in_port(0)

        if mul_POS_port is None:
            return

        negative = Negative(graph, {'name': name + '/Negate'}).create_node()

        mul.out_port(0).get_connection().set_source(negative.out_port(0))
        mul_POS_port.get_connection().set_destination(negative.in_port(0))


class EltwisesToSubtract(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [Negate]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('neg', {'type': 'Negative'}), # use multiply in addition
                ('neg_d', {}),
                ('add', {'type': 'Add'})
            ],
            edges=[
                ('neg', 'neg_d'),
                ('neg_d', 'add'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        neg = match['neg']
        add = match['add']

        name = add.soft_get('name', add.id)

        minuend_port = add.in_port(0).get_source() \
            if add.in_port(1).get_source().node.id == neg.id else add.in_port(1).get_source()
        subtrahned_port = neg.in_port(0).get_source()

        sub = Sub(graph, {'name': name + '/sub'}).create_node()

        add.out_port(0).get_connection().set_source(sub.out_port(0))
        minuend_port.connect(sub.in_port(0))
        subtrahned_port.connect(sub.in_port(1))
