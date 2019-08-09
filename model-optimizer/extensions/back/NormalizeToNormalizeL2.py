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

from extensions.back.EltwiseBroadcast import EltwiseBroadcast
from extensions.back.ElementwiseOpsToEltwiseOps import SimpleEltwiseToEltwiseOp
from extensions.ops.elementwise import Mul
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class NormalizeToNormalizeL2(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_before(self):
        return [SimpleEltwiseToEltwiseOp, EltwiseBroadcast]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('normalize', {'type': 'Normalize'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['normalize']
        assert node.in_port(0).data.get_shape().size in [2, 3, 4]
        assert node.has_valid('across_spatial')
        assert node.has_valid('channel_shared')
        assert node.has_valid('eps')

        if 'bin' in node.in_edge(1):
            del node.in_edge(1)['bin']

        weights = node.in_port(1).data.get_value()
        if node.channel_shared:
            node.in_port(1).data.set_value(np.array([weights[0]]))
        assert weights is not None
        if not np.all(weights == 1):
            mul = Mul(graph, {'name': node.name + '/Normalize_weights_multiplication'}).create_node()
            node.out_port(0).get_connection().set_source(mul.out_port(0))
            node.out_port(0).connect(mul.in_port(0))
            node.in_port(1).get_connection().get_source().connect(mul.in_port(1))
        node.in_port(1).disconnect()

        node['type'] = 'NormalizeL2'
        node['eps_mode'] = 'add'

        axes_val = np.array([1]) if not node.across_spatial else \
            np.arange(start=1, stop=node.in_port(0).data.get_shape().size)
        axes = Const(graph, {'value': axes_val}).create_node()
        node.in_port(1).connect(axes.out_port(0))

        del node['across_spatial']
        del node['channel_shared']
