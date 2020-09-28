"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
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
                ('neg', {'type': 'Negative'}),
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


class EltwisesToDiv(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('const', {'type': 'Const'}),
                ('const_d', {'value': lambda val: val is not None and np.all(val == -1)}),
                ('inv', {'type': 'Pow'}),
                ('inv_d', {}),
                ('mul', {'type': 'Multiply'})
            ],
            edges=[
                ('const', 'const_d'),
                ('const_d', 'inv', {'in': 1}),
                ('inv', 'inv_d'),
                ('inv_d', 'mul'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: [str, Node]):
        pow = match['inv']
        mul = match['mul']
        const = match['const']

        name = mul.soft_get('name', mul.id)

        devidend_port = mul.in_port(0).get_source() if mul.in_port(1).get_source().node.id == pow.id else mul.in_port(
            1).get_source()
        divider_port = pow.in_port(0).get_source() if pow.in_port(1).get_source().node.id == const.id else pow.in_port(
            1).get_source()

        div = Div(graph, {'name': name + '/div'}).create_node()

        mul.out_port(0).get_connection().set_source(div.out_port(0))
        devidend_port.connect(div.in_port(0))
        divider_port.connect(div.in_port(1))
