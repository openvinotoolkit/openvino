"""
 Copyright (C) 2017-2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class NonZero(Op):
    op = 'NonZero'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'infer': NonZero.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return
        input_value = node.in_node(0).value
        if input_value is not None:
            node.out_port(0).data.set_value(np.array(np.nonzero(input_value)))
        else:
            node.out_port(0).data.set_shape(int64_array([len(input_shape), *input_shape]))

