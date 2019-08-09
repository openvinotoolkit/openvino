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

from extensions.middle.AddIsCyclicAttribute import AddIsCyclicAttribute
from extensions.ops.TensorIterator_ops import TensorIteratorInput
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class SmartInputMatcher(MiddleReplacementPattern):
    """
    This pattern match partitioned inputs for TensorIterator in dynamic_rnn loops in TF.
    The structure of pattern without Data nodes between ops. Every node is named as op attribute of this node
    (data nodes is marked by (data)):
                                                        TensorArray
                                                        |          |
                                                        v          v                         Condition (data)
                                                   Flow(data)   Handle(data)--------------       |
                                                        |          |                      |      |
                                                        v          v                      v      v
    Value (data) -> StridedSlice () -> Range(0;1) -> TensorArrayScatter -> Enter -> TensorArrayRead
        |                                                  ^
        |__________________________________________________|
    """

    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [AddIsCyclicAttribute]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('TensorArray', dict(kind='op', op='TensorArrayV3')),
                ('TensorArray_handle', dict(kind='data')),
                ('TensorArray_flow', dict(kind='data')),
                ('Enter', dict(kind='op', op='Enter')),
                ('Enter_data', dict(kind='data')),

                ('stack', dict(kind='op', op='Const')),
                ('stack_data', dict(kind='data')),
                ('stack_1', dict(kind='op', op='Const')),
                ('stack_1_data', dict(kind='data')),
                ('stack_2', dict(kind='op', op='Const')),
                ('stack_2_data', dict(kind='data')),

                ('start', dict(kind='op', op='Const')),
                ('start_data', dict(kind='data')),

                ('delta', dict(kind='op', op='Const')),
                ('delta_data', dict(kind='data')),

                ('StridedSlice', dict(kind='op', op='StridedSlice')),
                ('StridedSlice_data', dict(kind='data')),
                ('range', dict(kind='op', op='Range')),
                ('range_data', dict(kind='data')),

                ('TensorArrayScatter', dict(kind='op', op='TensorArrayScatterV3')),
                ('TensorArrayScatter_data', dict(kind='data')),
                ('Enter_1', dict(kind='op', op='Enter')),
                ('Enter_1_data', dict(kind='data')),

                ('TensorArrayRead', dict(kind='op', op='TensorArrayReadV3')),
                ('TensorArrayRead_data', dict(kind='data')),

                ('Condition_data', dict(kind='data')),
            ],
            edges=[
                ('TensorArray', 'TensorArray_handle'),
                ('TensorArray', 'TensorArray_flow'),
                ('TensorArray_handle', 'Enter'),
                ('Enter', 'Enter_data'),

                ('stack', 'stack_data'),
                ('stack_1', 'stack_1_data'),
                ('stack_2', 'stack_2_data'),
                ('stack_data', 'StridedSlice', {'in': 1}),
                ('stack_1_data', 'StridedSlice', {'in': 2}),
                ('stack_2_data', 'StridedSlice', {'in': 3}),

                ('StridedSlice', 'StridedSlice_data'),
                ('StridedSlice_data', 'range', {'in': 1}),
                ('start', 'start_data'),
                ('delta', 'delta_data'),

                ('start_data', 'range', {'in': 0}),
                ('delta_data', 'range', {'in': 2}),
                ('range', 'range_data'),
                ('range_data', 'TensorArrayScatter'),

                ('TensorArray_handle', 'TensorArrayScatter'),
                ('TensorArray_flow', 'TensorArrayScatter'),
                ('TensorArrayScatter', 'TensorArrayScatter_data'),
                ('TensorArrayScatter_data', 'Enter_1'),
                ('Enter_1', 'Enter_1_data'),

                ('Enter_data', 'TensorArrayRead'),
                ('Enter_1_data', 'TensorArrayRead'),
                ('Condition_data', 'TensorArrayRead'),
                ('TensorArrayRead', 'TensorArrayRead_data'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SmartInputFind ===============')

        assert match['Enter_data'].value is not None
        assert match['stack_data']['value'][0] == 0 and match['stack_1_data']['value'][0] == 1 and \
               match['stack_2_data']['value'][0] == 1
        assert match['start_data']['value'] == 0 and match['delta_data']['value'] == 1

        ta_size_data = match['TensorArray'].in_node()
        ta_size = ta_size_data.in_node()
        value = match['TensorArrayScatter'].in_node(2)

        start, end = None, None
        if 0 in ta_size.in_nodes():
            shape = match['StridedSlice'].in_node(0).in_node(0)
            # Case when value for Strided slice is Const, not Shape
            if shape['kind'] == 'op' and shape['op'] == 'Const':
                start = 0
                end = shape.value[0]
                log.warning("You network cannot be reshaped since shapes of placeholders is a contants."
                            "Please, provide non-constant shapes. ")

        # Create input node with params
        # axis == 0 because in TensorArray we ALWAYS iterate over 0 axis, other params will be fill later (with
        # condition)
        input_node = TensorIteratorInput(graph, dict(axis=0, start=start, stride=None, part_size=None,
                                                     external_port_id=str(match['Enter_data'].value),
                                                     internal_layer_id=match['TensorArrayRead_data'].id,
                                                     name=match['TensorArrayRead'].name + '/TensorIteratorInput_'
                                                     ))
        input_node.create_node_with_data(inputs=[ta_size_data, value, match['Condition_data']],
                                         data_nodes=[match['TensorArrayRead_data']])
        # Delete useless nodes
        safe_nodes = ['TensorArrayRead_data', 'Condition', 'Condition_data']

        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)


class SimpleInputMatcher(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        from extensions.middle.DeleteNotExecutable import DeleteNotExecutable
        return [DeleteNotExecutable]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    """
    This pattern match simple inputs (without partitions) in while loops in TF (this inputs are set by Enter nodes).
    """

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('Enter', dict(kind='op', op='Enter')),
            ],
            edges=[
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SimpletInputFind ===============')

        input_node = TensorIteratorInput(graph, dict(external_port_id=None,
                                                     internal_layer_id=None,
                                                     name=match['Enter'].name + '/TensorIteratorInput_'
                                                     ))
        input_node.create_node_with_data(inputs=[match['Enter'].in_node()], data_nodes=[match['Enter'].out_node()])

        # Delete useless nodes
        graph.remove_nodes_from([match['Enter'].id])


class BackEdgeSimpleInputMatcher(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [SimpleInputMatcher]

    def run_before(self):
        from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('BackEdge', dict(kind='op', op='TensorIteratorBackEdge')),
            ],
            edges=[
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== SimpleBackEdgeInputFind ===============')

        assert len(match['BackEdge'].in_nodes()) == 3
        condition = match['BackEdge'].in_node(2)
        init_input = match['BackEdge'].in_node(0)
        cycle_input = match['BackEdge'].in_node(1)

        # We need to create new TensorItertorInput node only if this node doesn't exist already.
        if len(init_input.in_nodes()) == 0 or\
           (len(init_input.in_nodes()) == 1 and init_input.has_valid('value')):

            input_node = TensorIteratorInput(graph, dict(external_port_id=None,
                                                         internal_layer_id=None,
                                                         name=match['BackEdge'].name + '/TensorIteratorInput_'
                                                         ))

            # In case if data node has Constant producer
            if len(init_input.in_nodes()) == 1:
                graph.remove_edge(init_input.in_node(0).id, init_input.id)

            input_data_node = input_node.create_node_with_data(inputs=[init_input])
            input_data_node.shape = np.array(init_input.shape, dtype=np.int64)
            graph.remove_edges_from([(init_input.id, match['BackEdge'].id)])
            graph.add_edges_from([(input_data_node.id, match['BackEdge'].id, {'in': 0, 'out': 0})])
