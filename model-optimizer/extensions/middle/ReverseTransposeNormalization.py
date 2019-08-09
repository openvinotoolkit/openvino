"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class ReverseTransposeNormalization(MiddleReplacementPattern):
    enabled = True
    force_shape_inference = True

    def pattern(self):
        return dict(
            nodes=[('transpose', dict(type='Transpose', reverse_order=True))],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        node = match['transpose']
        assert len(node.in_nodes()) == 1
        order = np.arange(len(node.in_port(0).data.get_shape()))[::-1]
        const = Const(graph, {'value': order}).create_node()
        node.add_input_port(1, skip_if_exist=True)
        const.out_port(0).connect(node.in_port(1))
