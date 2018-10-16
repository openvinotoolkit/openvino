"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.ops.op import Op


class Eltwise(Op):
    op = 'Eltwise'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        operations = {
            'sum': ('Add', lambda a, b: a + b),
            'mul': ('Mul', lambda a, b: a * b),
            'max': ('Max', lambda a, b: np.max(a, b))
        }

        super().__init__(graph, {
            'type': 'Eltwise',  # a property of IE supported layer
            'op': operations[attrs['operation']][0],
            'infer': lambda node: eltwise_infer(node, operations[node.operation][1]),
        }, attrs)

    def supported_attrs(self):
        return ['operation']
