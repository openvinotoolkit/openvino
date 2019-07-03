"""
 Copyright (c) 2017-2019 Intel Corporation

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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ConstantFill(Op):
    """ Constant blob generation by broadcasting specified value to a given shape.

        It is assumed that there is no equivalent of this op in IE,
        so it is usually relevant to constant folding.
    """
    op = 'ConstantFill'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'input_as_shape': 1,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': __class__.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'input_as_shape',
            'fill_value'
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 1
        assert node.fill_value is not None
        assert node.input_as_shape

        shape = node.in_node(0).value
        assert shape is not None

        node.out_node(0).value = np.full(shape, node.fill_value, np.float32)
        node.out_node(0).shape = np.array(node.out_node(0).value.shape, dtype=np.int64)
