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

import logging as log

from extensions.ops.TensorIterator_ops import TensorIteratorOutput
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class SmartOutputMatcher(MiddleReplacementPattern):
    """
    This pattern match partitioned outputs for TensorIterator in dynamic_rnn loops in TF.
    The structure of pattern without Data nodes between ops. Every node is named as op attribute of this node
    (data nodes is marked by (data)):
        TensorArray
        |         |                                                                           Condition(data)
    Flow(data)  Handle(data)---------------------------------------------------------------     |
            |    |                                       |                                 |    |
            v    v                                       v                                 v    v
            Enter  ->  Merge -> Switch -> Exit -> TensorArraySize -> Range(0;1) -> TensorArrayGather
                                    |       |                                            ^
                                    |       |                                            |
                                    |        ---------------------------------------------
                                    |
                                    --------> Identity -> TensorArrayWrite -> NextIteration
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        from extensions.middle.TensorIteratorInput import SmartInputMatcher
        return [SmartInputMatcher]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('TensorArray', dict(kind='op', op='TensorArrayV3')),
                ('TensorArray_data', dict(kind='data')),
                ('TensorArray_flow_data', dict(kind='data')),
                ('TensorArrayGather', dict(kind='op', op='TensorArrayGatherV3')),
                ('TensorArrayGather_data', dict(kind='data')),
                ('range', dict(kind='op', op='Range')),
                ('range_data', dict(kind='data')),
                ('size', dict(kind='op', op='TensorArraySizeV3')),
                ('size_data', dict(kind='data')),
                ('start', dict(kind='op', op='Const')),
                ('start_data', dict(kind='data')),
                ('delta', dict(kind='op', op='Const')),
                ('delta_data', dict(kind='data')),
                ('TensorArrayWrite', dict(kind='op', op='TensorArrayWriteV3')),
                ('TensorArrayWrite_data', dict(kind='data')),
                ('NextIteration', dict(kind='op', op='NextIteration')),
                ('Condition_data', dict(kind='data')),
                ('Identity_2_data', dict(kind='data')),
                ('Identity_2', dict(kind='op', op='Identity')),
                ('Switch_2', dict(kind='op', op='Switch')),
                ('Switch_2_data', dict(kind='data')),
                ('Switch_2_data_exit', dict(kind='data')),
                ('Merge_2', dict(kind='op', op='Merge')),
                ('Merge_2_data', dict(kind='data')),
                ('Enter_2', dict(kind='op', op='Enter')),
                ('Enter_2_data', dict(kind='data')),
                ('WriteEnter', dict(kind='op', op='Enter')),
                ('WriteEnter_data', dict(kind='data')),
                ('Exit', dict(kind='op', op='Exit')),
                ('Exit_data', dict(kind='data')),
            ],
            edges=[
                ('TensorArray', 'TensorArray_data'),
                ('TensorArray', 'TensorArray_flow_data'),
                ('TensorArray_flow_data', 'Enter_2'),
                ('TensorArray_data', 'WriteEnter'),
                ('TensorArray_data', 'TensorArrayGather'),
                ('TensorArrayGather', 'TensorArrayGather_data'),
                ('TensorArray_data', 'size'),

                ('size', 'size_data'),
                ('start', 'start_data'),
                ('delta', 'delta_data'),

                ('size_data', 'range', {'in': 1}),
                ('start_data', 'range', {'in': 0}),
                ('delta_data', 'range', {'in': 2}),
                ('range', 'range_data'),
                ('range_data', 'TensorArrayGather'),

                ('Enter_2', 'Enter_2_data'),
                ('Enter_2_data', 'Merge_2'),
                ('Merge_2', 'Merge_2_data'),
                ('Merge_2_data', 'Switch_2'),
                ('Switch_2', 'Switch_2_data'),
                ('Switch_2', 'Switch_2_data_exit'),
                ('Switch_2_data', 'Identity_2'),
                ('Identity_2', 'Identity_2_data'),

                ('Switch_2_data_exit', 'Exit'),
                ('Exit', 'Exit_data'),
                ('Exit_data', 'size'),
                ('Exit_data', 'TensorArrayGather'),

                ('WriteEnter', 'WriteEnter_data'),
                ('WriteEnter_data', 'TensorArrayWrite', {'in': 0}),

                ('Identity_2_data', 'TensorArrayWrite', {'in': 3}),

                ('TensorArrayWrite', 'TensorArrayWrite_data'),
                ('TensorArrayWrite_data', 'NextIteration'),
                ('Condition_data', 'Switch_2'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SmartOutputFind ===============')

        assert match['WriteEnter_data'].value is not None
        assert match['start_data']['value'] == 0 and match['delta_data']['value'] == 1

        ta_size = match['TensorArray'].in_node()

        index = match['TensorArrayWrite'].in_node(1)
        value = match['TensorArrayWrite'].in_node(2)

        # axis == 0 because in TensorArray we ALWAYS iterate over 0 axis, other params will be fill later (with
        # condition)
        output = TensorIteratorOutput(graph, dict(axis=0, start=None, stride=None, part_size=None,
                                                  external_port_id=str(match['WriteEnter_data'].value),
                                                  internal_layer_id=value.id,
                                                  name=match['TensorArrayWrite'].name + '/TensorIteratorOutput_'
                                                  ))
        output.create_node_with_data(inputs=[ta_size, value, index],
                                     data_nodes=[match['TensorArrayGather_data']])

        # Delete useless nodes
        safe_nodes = ['TensorArrayGather_data', 'Condition_data']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)


class SimpleOutputMatcher(MiddleReplacementPattern):
    """
    This pattern match partitioned outputs for TensorIterator in dynamic_rnn loops in TF.
    The structure of pattern without Data nodes between ops. Every node is named as op attribute of this node
    (data nodes is marked by (data)):
        TensorArray
        |         |
    Flow(data)  Handle(data)------------------------------
            |    |                                       |
            v    v                                       v
            Enter  ->  Merge -> Switch -> Exit -> TensorArrayRead
                                    |
                                    |
                                    |
                                    |
                                    --------> Identity -> TensorArrayWrite -> NextIteration
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [SmartOutputMatcher]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        from extensions.middle.TensorIteratorCondition import LoopConditionMatcher
        return [TensorIteratorMerge, LoopConditionMatcher]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('TensorArray', dict(kind='op', op='TensorArrayV3')),
                ('TensorArray_data', dict(kind='data')),
                ('TensorArray_flow_data', dict(kind='data')),

                ('TensorArrayWrite', dict(kind='op', op='TensorArrayWriteV3')),
                ('TensorArrayWrite_data', dict(kind='data')),

                ('NextIteration', dict(kind='op', op='NextIteration')),
                ('NextIteration_data', dict(kind='data')),

                ('Condition_data', dict(kind='data')),

                ('Identity_2', dict(kind='op', op='Identity')),
                ('Identity_2_data', dict(kind='data')),

                ('Switch_2', dict(kind='op', op='Switch')),
                ('Switch_2_data', dict(kind='data')),
                ('Switch_2_data_exit', dict(kind='data')),

                ('Merge_2', dict(kind='op', op='Merge')),
                ('Merge_2_data', dict(kind='data')),

                ('Enter_2', dict(kind='op', op='Enter')),
                ('Enter_2_data', dict(kind='data')),

                ('WriteEnter', dict(kind='op', op='Enter')),
                ('WriteEnter_data', dict(kind='data')),

                ('Exit', dict(kind='op', op='Exit')),
                ('Exit_data', dict(kind='data')),
                #
                ('TensorArrayRead', dict(op='TensorArrayReadV3')),
                ('TensorArrayRead_data', dict(kind='data')),
            ],
            edges=[
                ('TensorArray', 'TensorArray_data'),
                ('TensorArray', 'TensorArray_flow_data'),
                ('TensorArray_flow_data', 'Enter_2'),
                ('TensorArray_data', 'WriteEnter'),


                ('Enter_2', 'Enter_2_data'),
                ('Enter_2_data', 'Merge_2'),
                ('Merge_2', 'Merge_2_data'),
                ('Merge_2_data', 'Switch_2'),
                ('Switch_2', 'Switch_2_data'),
                ('Switch_2', 'Switch_2_data_exit'),
                ('Switch_2_data', 'Identity_2'),
                ('Identity_2', 'Identity_2_data'),

                ('Switch_2_data_exit', 'Exit'),
                ('Exit', 'Exit_data'),
                ('Exit_data', 'TensorArrayRead'),

                ('WriteEnter', 'WriteEnter_data'),
                ('WriteEnter_data', 'TensorArrayWrite', {'in': 0}),

                ('Identity_2_data', 'TensorArrayWrite', {'in': 3}),
                #
                ('TensorArrayWrite', 'TensorArrayWrite_data'),
                ('TensorArrayWrite_data', 'NextIteration'),
                ('Condition_data', 'Switch_2'),
                #
                ('TensorArray_data', 'TensorArrayRead'),
                ('TensorArrayRead', 'TensorArrayRead_data'),
                ('NextIteration', 'NextIteration_data'),
                ('NextIteration_data', 'Merge_2'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SimpleOutputFind ===============')
        assert match['WriteEnter_data'].value is not None

        index = match['TensorArrayWrite'].in_node(1)
        value = match['TensorArrayWrite'].in_node(2)

        # axis == 0 because in TensorArray we ALWAYS iterate over 0 axis, other params will be fill later (with
        # condition)
        output = TensorIteratorOutput(graph, dict(
                                                  external_port_id=str(match['WriteEnter_data'].value),
                                                  internal_layer_id=value.id,
                                                  name=match['TensorArrayWrite'].name + '/TensorIteratorOutput_'
                                                  ))
        output.create_node_with_data(inputs=[value, index],
                                     data_nodes=[match['TensorArrayRead_data']])

        # Delete useless nodes
        safe_nodes = ['TensorArrayRead_data', 'Condition_data']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)
