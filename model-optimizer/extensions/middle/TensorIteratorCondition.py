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

import logging as log

import networkx as nx

from extensions.ops.TensorIterator_ops import TensorIteratorCondition
from mo.middle.replacement import MiddleReplacementPattern


class LoopConditionMatcher(MiddleReplacementPattern):
    """
    This pattern match condition for TensorIterator in while loops in TF.
    The structure of pattern without Data nodes between ops. Every node is named as op attribute of this node
    (data nodes is marked by (data)):
                                                                   Const----
                                                                            |
                                                                            v
    Const -> Enter -> Merge ---------------------> Switch -> Identity ->  Add -> NextIteration
                        |                              ^
                        ---> Less ----|                |
                                ^     |                |
    Maximum -> Minimum -> Enter-|     |                |
                 ^                    v                |
Shape -> StridedSlice -> Enter -|    LogicalAnd --> LoopCond (data)
                                v     ^                |
                        ---> Less ----|                |
                        |                              v
    Const -> Enter -> Merge ---------------------> Switch -> Identity ->  Add -> NextIteration
                                                                            ^
                                                                            |
                                                                   Const----
    """
    enabled = True

    @staticmethod
    def pattern():
        log.debug('+++++++++++++++ ConditionMatching ++++++++++++++++')
        return dict(
            nodes=[
                ('Enter_1_less', dict(kind='op', op='Enter')),
                ('Strided_slice', dict(kind='op', op='StridedSlice')),
                ('Strided_slice_data', dict(kind='data')),
                ('Enter_1_less_data', dict(kind='data')),

                ('Less_1', dict(kind='op', op='Less')),
                ('Merge_1', dict(kind='op', op='Merge')),
                ('Merge_1_data', dict(kind='data')),
                ('Less_1_data', dict(kind='data')),

                ('Less_2', dict(kind='op', op='Less')),
                ('Merge_2', dict(kind='op', op='Merge')),
                ('Merge_2_data', dict(kind='data')),
                ('Less_2_data', dict(kind='data')),

                ('Enter_2_less', dict(kind='op', op='Enter')),
                ('Enter_2_less_data', dict(kind='data')),
                ('minimum', dict(kind='op', op='Minimum')),
                ('minimum_data', dict(kind='data')),
                ('Maximum',  dict(kind='op', op='Maximum')),
                ('Maximum_data', dict(kind='data')),

                ('and', dict(kind='op', op='LogicalAnd')),
                ('and_data', dict(kind='data')),
                ('loop_cond', dict(kind='op', op='LoopCond')),
                ('loop_cond_data', dict(kind='data')),

                ('init_1', dict(kind='op', op='Const')),
                ('init_1_data',  dict(kind='data')),
                ('Enter_1', dict(kind='op', op='Enter')),
                ('Enter_1_data',  dict(kind='data')),

                ('init_2', dict(kind='op', op='Const')),
                ('init_2_data', dict(kind='data')),
                ('Enter_2', dict(kind='op', op='Enter')),
                ('Enter_2_data', dict(kind='data')),

                ('Switch_1', dict(kind='op', op='Switch')),
                ('Switch_1_data', dict(kind='data')),
                ('Identity_1', dict(kind='op', op='Identity')),
                ('Identity_1_data', dict(kind='data')),
                ('add_1', dict(kind='op', op='Add')),
                ('add_1_y',  dict(kind='op', op='Const')),
                ('add_1_y_data', dict(kind='data')),
                ('add_1_data', dict(kind='data')),
                ('NextIteration_1', dict(kind='op', op='NextIteration')),

                ('Switch_2', dict(kind='op', op='Switch')),
                ('Switch_2_data', dict(kind='data')),
                ('Identity_2', dict(kind='op', op='Identity')),
                ('Identity_2_data', dict(kind='data')),
                ('add_2', dict(kind='op', op='Add')),
                ('add_2_y', dict(kind='op', op='Const')),
                ('add_2_y_data', dict(kind='data')),
                ('add_2_data', dict(kind='data')),
                ('NextIteration_2', dict(kind='op', op='NextIteration')),

            ],
            edges=[
                ('Strided_slice', 'Strided_slice_data'),
                ('Strided_slice_data', 'Enter_1_less'),
                ('Strided_slice_data', 'minimum'),
                ('Enter_1_less', 'Enter_1_less_data'),
                ('Enter_1_less_data', 'Less_1'),
                ('Less_1', 'Less_1_data'),
                ('Less_1_data', 'and'),

                ('and', 'and_data'),
                ('and_data', 'loop_cond'),
                ('loop_cond', 'loop_cond_data'),
                ('loop_cond_data', 'Switch_1'),
                ('loop_cond_data', 'Switch_2'),

                ('init_1', 'init_1_data'),
                ('init_1_data', 'Enter_1'),
                ('Enter_1', 'Enter_1_data'),
                ('Enter_1_data', 'Merge_1'),
                ('Merge_1', 'Merge_1_data'),
                ('Merge_1_data', 'Less_1'),

                ('Merge_1_data', 'Switch_1'),
                ('Switch_1', 'Switch_1_data'),
                ('Switch_1_data', 'Identity_1'),
                ('Identity_1', 'Identity_1_data'),
                ('Identity_1_data', 'add_1'),
                ('add_1_y', 'add_1_y_data'),
                ('add_1_y_data', 'add_1'),
                ('add_1', 'add_1_data'),
                ('add_1_data', 'NextIteration_1'),

                ('Merge_2_data', 'Switch_2'),
                ('Switch_2', 'Switch_2_data'),
                ('Switch_2_data', 'Identity_2'),
                ('Identity_2', 'Identity_2_data'),
                ('Identity_2_data', 'add_2'),
                ('add_2_y', 'add_2_y_data'),
                ('add_2_y_data', 'add_2'),
                ('add_2', 'add_2_data'),
                ('add_2_data', 'NextIteration_2'),

                ('Maximum', 'Maximum_data'),
                ('Maximum_data', 'minimum'),
                ('minimum', 'minimum_data'),
                ('minimum_data', 'Enter_2_less'),
                ('Enter_2_less', 'Enter_2_less_data'),
                ('Enter_2_less_data', 'Less_2'),

                ('init_2', 'init_2_data'),
                ('init_2_data', 'Enter_2'),
                ('Enter_2', 'Enter_2_data'),
                ('Enter_2_data', 'Merge_2'),

                ('Merge_2', 'Merge_2_data'),
                ('Merge_2_data', 'Less_2'),
                ('Less_2', 'Less_2_data'),
                ('Less_2_data', 'and'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        log.debug('================== ConditionFind ===============')

        #init_1
        init_1 = match['init_1_data'].value
        assert init_1 is not None
        init_1 = int(init_1)

        #init_2
        init_2 = match['init_2_data'].value
        assert init_2 is not None
        init_2 = int(init_2)

        #step_1
        assert match['add_1_y_data'].value is not None
        step_1 = int(match['add_1_y_data'].value)

        #step_2
        assert match['add_2_y_data'].value is not None
        step_2 = int(match['add_2_y_data'].value)

        match['loop_cond_data'].value = None
        match['Identity_2_data'].value = None

        # Create condition node and delete all useless nodes from condition pattern
        condition_attrs = dict(time=dict(init=init_2, step=step_2), iter=dict(init=init_1, step=step_1), \
                               name=match['loop_cond'].name + '/TensorIteratorCondition_')
        condition = TensorIteratorCondition(graph, attrs=condition_attrs)
        condition.create_node_with_data(inputs=[match['Strided_slice_data'], match['minimum_data']],
                                        data_nodes=[match['loop_cond_data'], match['Identity_2_data']])

        # Delete useless nodes
        safe_nodes = ['loop_cond_data', 'Identity_2_data', 'Strided_slice', 'Strided_slice_data',
                      'Maximum', 'Maximum_data', 'minimum', 'minimum_data']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)
