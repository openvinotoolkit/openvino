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
from extensions.middle.PartialInfer import PartialInfer
from extensions.middle.TensorIteratorInput import SmartInputMatcher
from extensions.ops.select import Select
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class SwitchMergeMiddleReplacer(MiddleReplacementPattern):
    """
    Optimization for case, when combination of Switches have one common condition and can be expressed as Select node.
    """
    enabled = True

    def run_before(self):
        return [SmartInputMatcher]

    def run_after(self):
        return [PartialInfer]

    def pattern(self):
        return dict(
            nodes=[('Merge', dict(kind='op', op='Merge')),

                   ('Switch_2_input', dict(kind='data')),

                   ('Switch_2', dict(kind='op', op='Switch')),
                   ('Switch_2_data', dict(kind='data')),

                   ('some_op', dict(kind='op')),
                   ('some_op_data', dict(kind='data')),

                   ('Switch', dict(kind='op', op='Switch')),
                   ('Switch_data', dict(kind='data')),

                   ('Switch_1', dict(kind='op', op='Switch')),
                   ('Switch_1_data', dict(kind='data')),

                   ('cond_data', dict(kind='data')),

                   ('identity', dict(kind='op', op='Identity')),
                   ('identity_data', dict(kind='data')),
                   ],
            edges=[
                   ('Switch_2_input', 'Switch_2', {'in': 0}),
                   ('Switch_2', 'Switch_2_data', {'out': 1}),
                   ('Switch_2_data', 'Merge'),

                   ('cond_data', 'Switch_2', {'in': 1}),
                   ('cond_data', 'Switch_1', {'in': 1}),
                   ('cond_data', 'Switch', {'in': 1}),

                   ('Switch_1', 'Switch_1_data', {'out': 0}),

                   ('Switch', 'Switch_data', {'out': 0}),

                   ('Switch_1_data', 'some_op', {'in': 1}),
                   ('Switch_data', 'some_op', {'in': 0}),

                   ('some_op', 'some_op_data'),
                   ('some_op_data', 'identity'),
                   ('identity', 'identity_data'),
                   ('identity_data', 'Merge'),
                   ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        condition = match['cond_data']
        true_value = match['Switch_2_input']
        false_value = match['identity_data']

        select = Select(graph, dict(name=match['Merge'].name + '/Select/',
                                    format='tf')).create_node(inputs=[condition, true_value, false_value])

        match['Merge'].out_port(0).get_connection().set_source(select.out_port(0))

        # Reconnect inputs to some_op
        op = match['some_op']
        assert 1 in op.in_ports() and 0 in op.in_ports()

        op.in_port(0).disconnect()
        op.in_port(1).disconnect()
        match['Switch'].in_port(0).get_connection().set_destination(op.in_port(0))
        match['Switch_1'].in_port(0).get_connection().set_destination(op.in_port(1))

        graph.remove_nodes_from(nodes=[match['Switch_1'].id, match['Switch'].id, match['Switch_2'].id, match['Merge'].id])
