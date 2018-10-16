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


class Power(Op):
    enabled = False

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'type': 'Power',
            'op': 'Power',
            'power': 1,
            'scale': 1,
            'shift': 0,
            'infer': lambda node: eltwise_infer(node, lambda a: np.power(a * node.scale + node.shift, node.power)),
        }, attrs)

    def supported_attrs(self):
        """
        List of attributes that can/should be set by a client.
        """
        return ['power', 'scale', 'shift']
