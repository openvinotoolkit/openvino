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

from mo.ops.op import Op
from mo.front.common.partial_infer.eltwise import eltwise_infer


class LinOp(Op):
    enabled = False
    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'can_be_bias': True,
            'can_be_fused': True,
            'type': 'Eltwise',
            'infer': None,
        }, attrs)


class Add(LinOp):
    enabled = False
    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        attrs.update({'op': 'Add', 'operation': 'sum', 'infer': lambda node: eltwise_infer(node, lambda a, b: a + b)})
        super().__init__(graph, attrs)


class Mul(LinOp):
    enabled = False
    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        attrs.update({'op': 'Mul', 'operation': 'mul', 'infer': lambda node: eltwise_infer(node, lambda a, b: a*b)})
        super().__init__(graph, attrs)
