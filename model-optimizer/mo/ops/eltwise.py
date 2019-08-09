"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class Eltwise(Op):
    op = 'Eltwise'

    def __init__(self, graph: Graph, attrs: dict):
        operations = {
            'sum': ('Add', lambda a, b: a + b),
            'mul': ('Mul', lambda a, b: a * b),
            'max': ('Max', lambda a, b: np.maximum(a, b)),
            'pow': ('Pow', lambda a, b: np.power(a, b)),
            'greater_equal': ('GreaterEqual', lambda a, b: a >= b),
            'less': ('Less', lambda a, b: a < b),
            'less_equal': ('LessEqual', lambda a, b: a <= b),
            'greater': ('Greater', lambda a, b: a > b),
            'greater_equal': ('GreaterEqual', lambda a, b: a >= b),
            'equal': ('Equal', lambda a, b: a == b),
            'floor_mod': ('FloorMod', lambda a, b: a % b),
            'not_equal': ('NotEqual', lambda a, b: a != b),
            'logical_or': ('LogicalOr', lambda a, b: bool(a) or bool(b)),
            'logical_and': ('LogicalAnd', lambda a, b: bool(a) and bool(b)),
            'logical_xor': ('LogicalXor', lambda a, b: bool(a) ^ bool(b)),
            'log': ('Log', lambda x: np.log(x)),
        }

        super().__init__(graph, {
            'type': __class__.op,
            'op': operations[attrs['operation']][0],
            'infer': lambda node: eltwise_infer(node, operations[node.operation][1]),
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['operation']
