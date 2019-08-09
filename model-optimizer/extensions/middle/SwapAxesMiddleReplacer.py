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

from extensions.ops.transpose import Transpose
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class SwapAxisMiddleReplacer(MiddleReplacementPattern):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', op='SwapAxis'))],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        swapaxis = match['op']
        assert len(swapaxis.in_ports()) == 1
        assert swapaxis.has_and_set('order')
        order = swapaxis.order

        swapaxis.add_input_port(1)
        const = Const(graph, {'value': order}).create_node()
        const.out_port(0).connect(swapaxis.in_port(1))

        Transpose.update_node_stat(swapaxis, {'need_shape_inference': True})

        del swapaxis['order']
