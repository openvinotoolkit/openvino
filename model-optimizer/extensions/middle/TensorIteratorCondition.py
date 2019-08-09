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

import numpy as np

from extensions.middle.PartialInfer import PartialInfer
from extensions.middle.TensorIterator_utils import delete_selects_from
from extensions.ops.TensorIterator_ops import TensorIteratorCondition, TensorIteratorBackEdge
from extensions.ops.elementwise import Mul
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


def make_nodes_1D(nodes: list):
    """
    Reshape every node from nodes from 0D to 1D (nodes should have shape attribute).
    """
    for node in nodes:
        assert node.shape is None or len(node.shape) == 0
        node.shape = np.array([1], dtype=np.int64)
        if node.value is not None:
            node.value = np.reshape(node.value, node.shape)


def looking_for_op_in_list(nodes: list, op: str):
    for node in nodes:
        if node.has_valid('op') and node.op == op:
            return node

    return None


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
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return []

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

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
                ('minimum_data', dict(kind='data')),

                ('and', dict(kind='op', op='LogicalAnd')),
                ('and_data', dict(kind='data')),
                ('loop_cond', dict(kind='op', op='LoopCond')),
                ('loop_cond_data', dict(kind='data')),

                ('init_1', dict(kind='op', op='Const')),
                ('init_1_data', dict(kind='data')),
                ('Enter_1', dict(kind='op', op='Enter')),
                ('Enter_1_data', dict(kind='data')),

                ('init_2', dict(kind='op', op='Const')),
                ('init_2_data', dict(kind='data')),
                ('Enter_2', dict(kind='op', op='Enter')),
                ('Enter_2_data', dict(kind='data')),

                ('Switch_1', dict(kind='op', op='Switch')),
                ('Switch_1_data', dict(kind='data')),
                ('Identity_1', dict(kind='op', op='Identity')),
                ('Identity_1_data', dict(kind='data')),
                ('add_1', dict(kind='op', op='Add')),
                ('add_1_y', dict(kind='op', op='Const')),
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
    def looking_for_iteration_counter(graph: Graph, match: dict):
        types = ['TensorIteratorInput', 'TensorIteratorOutput']
        candidates = np.array([match['Identity_1_data'], match['Identity_2_data']])
        results = np.array([False for i in range(len(candidates))])
        for i, candidat in enumerate(candidates):
            for node in candidat.out_nodes():
                if node['op'] in types:
                    results[i] = True
        assert not np.all(results)
        assert sum(results) == 1
        return candidates[results == True][0]

    @staticmethod
    def check_dynamic_seq_len(graph: Graph, match: dict):
        """
        Cycle is dynamic if at least one of the boundaries isn't constant OR this boundaries is different from tensor
        shape.
        """
        dynamic_seq_len = match['Enter_1_less_data'].value is None or match['Enter_2_less_data'].value is None or \
                                not np.array_equal(match['Enter_1_less_data'].value, match['Enter_2_less_data'].value)

        return dynamic_seq_len

    def replace_pattern(self, graph: Graph, match: dict):
        log.debug('================== ConditionFind ===============')
        # init_1
        init_1 = match['init_1_data'].value
        assert init_1 is not None
        init_1 = int(init_1)

        # init_2
        init_2 = match['init_2_data'].value
        assert init_2 is not None
        init_2 = int(init_2)

        # step_1
        assert match['add_1_y_data'].value is not None
        step_1 = int(match['add_1_y_data'].value)

        # step_2
        assert match['add_2_y_data'].value is not None
        step_2 = int(match['add_2_y_data'].value)

        dynamic_seq_len = self.check_dynamic_seq_len(graph, match)

        # Create condition node and delete all useless nodes from condition pattern
        loop_condition = match['loop_cond_data']
        iterator_data = self.looking_for_iteration_counter(graph, match)

        condition_attrs = dict(time=dict(init=init_2, step=step_2), iter=dict(init=init_1, step=step_1),
                               name=match['loop_cond'].name + '/TensorIteratorCondition_')
        condition = TensorIteratorCondition(graph, attrs=condition_attrs)
        condition_data = condition.create_node_with_data(inputs=[match['Strided_slice_data'], match['minimum_data']],
                                                         data_nodes=[loop_condition, iterator_data])

        safe_nodes = ['loop_cond_data', 'Identity_1_data', 'Identity_2_data', 'Strided_slice', 'Strided_slice_data',
                      'minimum', 'minimum_data']

        identity_ops = [n.op for n in iterator_data.out_nodes()]
        if 'GreaterEqual' in identity_ops:
            greater_equal_id = [n.id for n in iterator_data.out_nodes() if n.op == 'GreaterEqual'][0]

            if dynamic_seq_len:
                # Add BackEdge for time iterator node
                backedge = TensorIteratorBackEdge(graph, dict(name='/TimeIterator/TensorIteratorBackEdge_'))
                backedge_data = backedge.create_node_with_data(inputs=[match['init_2_data'], match['add_2_data'],
                                                               condition_data[0]],)

                graph.remove_edge(match['add_2'].in_node(0).id, match['add_2'].id)
                graph.add_edge(backedge_data.id, match['add_2'].id, **{'in': 0})

                graph.remove_edge(iterator_data.id, greater_equal_id)
                graph.add_edge(backedge_data.id, greater_equal_id, **{'in': 0})

                # nodes for time iterator
                safe_nodes += ['init_2_data', 'init_2', 'Identity_2_data', 'add_2_data', 'add_2', 'add_2_y', 'add_2_y_data']

                # Manually reshape all iterator nodes (for time) from 0D to 1D
                iterator_data_nodes = [backedge_data, match['add_2_data'], match['add_2_y_data'], match['add_2_y'],
                                       match['init_2_data'], match['init_2']]
                make_nodes_1D(iterator_data_nodes)
            else:
                # Delete Selects from this cycle to make it not dynamic:
                greater_equal_idxs = [n.id for n in iterator_data.out_nodes() if n.op == 'GreaterEqual']
                delete_selects_from(graph, greater_equal_idxs)

        # Delete useless nodes
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)


class SimpleConditionMatcher(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [LoopConditionMatcher]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        log.debug('+++++++++++++++ SimpleConditionMatching ++++++++++++++++')
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

                ('loop_cond', dict(kind='op', op='LoopCond')),
                ('loop_cond_data', dict(kind='data')),

                ('init_1', dict(kind='op', op='Const')),
                ('init_1_data', dict(kind='data')),
                ('Enter_1', dict(kind='op', op='Enter')),
                ('Enter_1_data', dict(kind='data')),

                ('Switch_1', dict(kind='op', op='Switch')),
                ('Switch_1_data', dict(kind='data')),
                ('Identity_1', dict(kind='op', op='Identity')),
                ('Identity_1_data', dict(kind='data')),
                ('add_1', dict(kind='op', op='Add')),
                ('add_1_y', dict(kind='op', op='Const')),
                ('add_1_y_data', dict(kind='data')),
                ('add_1_data', dict(kind='data')),
                ('NextIteration_1', dict(kind='op', op='NextIteration')),
            ],
            edges=[
                ('Strided_slice', 'Strided_slice_data'),
                ('Strided_slice_data', 'Enter_1_less'),
                ('Enter_1_less', 'Enter_1_less_data'),
                ('Enter_1_less_data', 'Less_1'),
                ('Less_1', 'Less_1_data'),
                ('Less_1_data', 'loop_cond'),

                ('loop_cond', 'loop_cond_data'),
                ('loop_cond_data', 'Switch_1'),

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

            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SimpleConditionFind ===============')
        # init_1
        init_1 = match['init_1_data'].value
        assert init_1 is not None
        init_1 = int(init_1)

        # step_1
        assert match['add_1_y_data'].value is not None
        step_1 = int(match['add_1_y_data'].value)

        match['loop_cond_data'].value = None

        # Create condition node and delete all useless nodes from condition pattern
        condition_attrs = dict(iter=dict(init=init_1, step=step_1),
                               name=match['loop_cond'].name + '/TensorIteratorCondition_')
        condition = TensorIteratorCondition(graph, attrs=condition_attrs)
        condition.create_node_with_data(inputs=[match['Strided_slice_data']],
                                        data_nodes=[match['loop_cond_data'], match['Identity_1_data']])

        # Delete useless nodes
        safe_nodes = ['loop_cond_data', 'Identity_1_data', 'Strided_slice', 'Strided_slice_data']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)


class DynamicDecoderConditionMatcher(MiddleReplacementPattern):
    """
        This pattern match condition for dynamic decoder and create TensorIteratorCondition node instead of it.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [SimpleConditionMatcher]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        log.debug('+++++++++++++++ DynamicDecoderConditionMatching ++++++++++++++++')
        return dict(
            nodes=[
                ('loop_cond', dict(kind='op', op='LoopCond')),
                ('loop_cond_data', dict(kind='data')),

                ('logical_not', dict(kind='op', op='Not')),
                ('logical_not_data', dict(kind='data')),

                ('all', dict(kind='op', op='ReduceAnd')),
                ('all_data', dict(kind='data')),

                ('Merge_16', dict(kind='op', op='Merge')),
                ('merge_16_data', dict(kind='data')),

                ('NextIteration_16', dict(kind='op', op='NextIteration')),
                ('nextIteration_data', dict(kind='data')),

                ('Switch', dict(kind='op', op='Switch')),
                ('switch_data', dict(kind='data')),

                ('Identity', dict(kind='op', op='Identity')),
                ('identity_data', dict(kind='data')),

                ('add', dict(kind='op', op='Add')),
                ('add_data', dict(kind='data')),

                ('Less_enter',  dict(kind='op', op='Enter')),
                ('Less_enter_data', dict(kind='data')),

                ('And', dict(kind='op', op='LogicalAnd')),
                ('And_data', dict(kind='data')),

                ('Less',  dict(kind='op', op='Less')),
                ('Less_data', dict(kind='data')),

                ('TensorIteratorOutput', dict(kind='op', op='TensorIteratorOutput')),
                ('TensorIteratorOutput_1', dict(kind='op', op='TensorIteratorOutput')),
            ],
            edges=[
                ('NextIteration_16', 'nextIteration_data'),
                ('nextIteration_data', 'Merge_16'),
                ('Merge_16', 'merge_16_data'),
                ('merge_16_data', 'all'),
                ('all', 'all_data'),
                ('all_data', 'logical_not'),
                ('logical_not', 'logical_not_data'),

                ('Less_enter', 'Less_enter_data'),
                ('Less_enter_data', 'Less'),

                ('Less', 'Less_data'),
                ('Less_data', 'And'),

                ('logical_not_data', 'And'),
                ('And', 'And_data'),
                ('And_data', 'loop_cond'),

                ('loop_cond', 'loop_cond_data'),

                ('loop_cond_data', 'Switch'),

                ('Switch', 'switch_data'),

                ('switch_data', 'Identity'),

                ('Identity', 'identity_data'),

                ('identity_data', 'add'),
                ('add', 'add_data'),

                ('identity_data', 'TensorIteratorOutput'),
                ('identity_data', 'TensorIteratorOutput_1'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Create condition node and delete all useless nodes (like Switch/Merge/Identity) from condition pattern
        """
        log.debug('================== DynamicDecoderConditionFind  ==================')
        # Create and connect condition node for dynamic decoder in TF
        loop_condiiton = match['loop_cond_data']
        iterator_data = match['identity_data']

        condition_attrs = dict(name=match['loop_cond'].name + '/TensorIteratorCondition_')
        condition = TensorIteratorCondition(graph, attrs=condition_attrs)
        condition.create_node_with_data(inputs=[match['Less_enter'].in_node()],
                                        data_nodes=[loop_condiiton, iterator_data])

        # Delete useless nodes
        safe_nodes = ['loop_cond_data', 'identity_data', 'TensorIteratorOutput', 'TensorIteratorOutput_1']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)
