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

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.ops.strided_slice import StridedSlice


class PriorboxMutation(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_shape_inference = True

    def run_before(self):
        return [ForceStrictPrecision]

    def pattern(self):
        return dict(
            nodes=[
                ('pb', {'type': lambda node_type: node_type in ['PriorBox', 'PriorBoxClustered']})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pb']
        assert len(node.in_ports()) == 2

        begin = Const(graph, {'value': np.array([2])}).create_node()
        end = Const(graph, {'value': np.array([4])}).create_node()
        stride = Const(graph, {'value': np.array([1])}).create_node()

        shape_0 = Shape(graph, {'name': node.name + '/0_port', 'stop_value_propagation': True}).create_node()
        ss_0 = StridedSlice(graph, {'name': node.name + '/ss_0_port', 'begin_mask': np.array([1]),
                                    'end_mask': np.array([0]), 'new_axis_mask': np.array([0]),
                                    'shrink_axis_mask': np.array([0]),
                                    'ellipsis_mask': np.array([0])}).create_node()

        shape_0.out_port(0).connect(ss_0.in_port(0))
        begin.out_port(0).connect(ss_0.in_port(1))
        end.out_port(0).connect(ss_0.in_port(2))
        stride.out_port(0).connect(ss_0.in_port(3))

        source = node.in_port(0).get_connection().get_source()
        node.in_port(0).disconnect()
        source.connect(shape_0.in_port(0))
        ss_0.out_port(0).connect(node.in_port(0))

        shape_1 = Shape(graph, {'name': node.name + '/1_port', 'stop_value_propagation': True}).create_node()
        ss_1 = StridedSlice(graph, {'name': node.name + '/ss_1_port', 'begin_mask': np.array([1]),
                                    'end_mask': np.array([0]), 'new_axis_mask': np.array([0]),
                                    'shrink_axis_mask': np.array([0]),
                                    'ellipsis_mask': np.array([0])}).create_node()

        shape_1.out_port(0).connect(ss_1.in_port(0))
        begin.out_port(0).connect(ss_1.in_port(1))
        end.out_port(0).connect(ss_1.in_port(2))
        stride.out_port(0).connect(ss_1.in_port(3))

        source = node.in_port(1).get_connection().get_source()
        node.in_port(1).disconnect()
        source.connect(shape_1.in_port(0))
        ss_1.out_port(0).connect(node.in_port(1))

        ss_0['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}
        ss_1['force_precision_in_ports'] = {1: 'int64', 2: 'int64', 3: 'int64'}

