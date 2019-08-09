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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class PadNormalize(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('pad', dict(kind='op', type='Pad'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pad']

        pb = node.pads[:, 0]
        pe = node.pads[:, 1]
        pm = node.mode

        pads_begin = Const(graph, {'value': np.array(pb)}).create_node()
        node.add_input_port(1, skip_if_exist=True)
        node.in_port(1).connect(pads_begin.out_port(0))
        pads_begin.infer(pads_begin)

        pads_end = Const(graph, {'value': np.array(pe)}).create_node()
        node.add_input_port(2, skip_if_exist=True)
        node.in_port(2).connect(pads_end.out_port(0))
        pads_end.infer(pads_end)

        del node['pads']

        if node.has_valid('fill_value') and pm == 'constant':
            pv = node.fill_value
            pad_value = Const(graph, {'value': np.array(pv)}).create_node()
            node.add_input_port(3, skip_if_exist=True)
            node.in_port(3).connect(pad_value.out_port(0))
            pad_value.infer(pad_value)

        del node['fill_value']

        node['need_shape_inference'] = False
