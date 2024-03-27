# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.middle.AddIsCyclicAttribute import AddIsCyclicAttribute
from openvino.tools.mo.ops.TensorIterator_ops import TensorIteratorInput
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


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
        from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
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
                log.warning("Your network cannot be reshaped since shapes of placeholders are constants. "
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
        from openvino.tools.mo.middle.DeleteNotExecutable import DeleteNotExecutable
        return [DeleteNotExecutable]

    def run_before(self):
        from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
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
        from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
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
        if (len(init_input.in_nodes()) == 0 or \
                (len(init_input.in_nodes()) == 1 and init_input.has_valid('value') and
                 init_input.in_node(0).soft_get('op') != 'TensorIteratorInput')):

            input_node = TensorIteratorInput(graph, dict(external_port_id=None,
                                                         internal_layer_id=None,
                                                         name=match['BackEdge'].name + '/TensorIteratorInput_'
                                                         ))

            # In case if data node has Constant producer
            if len(init_input.in_nodes()) == 1:
                graph.remove_edge(init_input.in_node(0).id, init_input.id)

            input_data_node = input_node.create_node_with_data(inputs=[init_input])
            input_data_node.shape = int64_array(init_input.shape)
            graph.remove_edges_from([(init_input.id, match['BackEdge'].id)])
            graph.add_edges_from([(input_data_node.id, match['BackEdge'].id, {'in': 0, 'out': 0})])


class SmartMatcherInputSlicingWithGather(MiddleReplacementPattern):
    r"""
    The transformation matches a sub-graph where input tensor is consequently sliced along some axis
    for each time step (or index) inside TensorFlow 1.x while_loop operation.
    In the original graph StridedSlice with non-constant begin and end attributes performs this slicing.
    NonConstBeginStridedSliceReplacement, a front transformation, replaces this StridedSlice with Gather operation
    after which the following sub-graph is obtained (Note: no data node is displayed):

                              NextIteration <------- Add <--- Time Step
                                |                     /\
                               \/                     |
    InitTime ----> Enter --> Merge ---> Switch ---> Identity ------
                              |          /\                        |
                             \/          |                         |
               MaxTime ---> Less ---> LoopCond                 Unsqueeze (axis=0)
                                         |                         |
                                        \/                        \/
    Input ---> Enter ----> Merge ---> Switch ---> Identity ---> Gather ---> Squeeze --> Ops (Need Slice at i-th time)
                            /\                       |            /\           /\
                            |                       \/            |----Axis----|
                            -------------------- NextIteration

    Some part of the sub-graph above is replaced with TensorIteratorInput and the following graph is obtained
    after the transformation:

                              NextIteration <------- Add <--- Time Step
                                |                     /\
                               \/                     |
    InitTime ----> Enter --> Merge ---> Switch ---> Identity ------|
                              |          /\                        |
                             \/          |                         |
               MaxTime ---> Less ---> LoopCond                     |
                 |                                                 |
                 |       |-----------------------------------------
                \/      \/
    Input --> TensorIteratorInput(InitTime, TimeStep, Axis) ---> Ops (Need Slice at i-th time)

    Details about TensorIterator (inputs, outputs, and attributes) will be finally used by TensorIteratorMerge
    transformation during construction of TensorIterator operation.
    """

    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return [AddIsCyclicAttribute]

    def run_before(self):
        from openvino.tools.mo.middle.TensorIteratorBackEdge import BackEdgesMatching
        from openvino.tools.mo.middle.TensorIteratorCondition import LoopConditionMatcher
        return [BackEdgesMatching, LoopConditionMatcher]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                # LoopCond node and related Condition node
                ('EnterMaxIndex', dict(kind='op', op='Enter')),
                ('EnterMaxIndexData', dict(kind='data')),
                ('Less', dict(kind='op', op='Less')),
                ('LessData', dict(kind='data')),
                ('LoopCond', dict(kind='op', op='LoopCond')),
                ('LoopCondData', dict(kind='data')),

                # a list of Input specific nodes
                ('EnterInput', dict(kind='op', op='Enter')),
                ('EnterInputData', dict(kind='data')),
                ('MergeInput', dict(kind='op', op='Merge')),
                ('MergeInputData', dict(kind='data')),
                ('SwitchInput', dict(kind='op', op='Switch')),
                ('SwitchInputData', dict(kind='data')),
                ('IdentityInput', dict(kind='op', op='Identity')),
                ('IdentityInputData', dict(kind='data')),
                ('NextIterationInput', dict(kind='op', op='NextIteration')),

                # a list of Index specific nodes
                ('InitIndex', dict(kind='op', op='Const')),
                ('InitIndexData', dict(kind='data')),
                ('EnterIndex', dict(kind='op', op='Enter')),
                ('EnterIndexData', dict(kind='data')),
                ('MergeIndex', dict(kind='op', op='Merge')),
                ('MergeIndexData', dict(kind='data')),
                ('SwitchIndex', dict(kind='op', op='Switch')),
                ('SwitchIndexData', dict(kind='data')),
                ('IdentityIndex', dict(kind='op', op='Identity')),
                ('IdentityIndexData', dict(kind='data')),
                ('UnsqueezeIndex', dict(kind='op', op='Unsqueeze')),
                ('UnsqueezeIndexData', dict(kind='data')),
                ('AddIndex', dict(kind='op', op='Add')),
                ('AddIndexData', dict(kind='data')),
                ('NextIterationIndex', dict(kind='op', op='NextIteration')),
                ('IndexDelta', dict(kind='op', op='Const')),
                ('IndexDeltaData', dict(kind='data')),

                # a list of nodes responsible for slicing
                ('Axis', dict(kind='op', op='Const')),
                ('AxisData', dict(kind='data')),
                ('Gather', dict(kind='op', op='Gather')),
                ('GatherData', dict(kind='data')),
                ('SqueezeSlice', dict(kind='op', op='Squeeze')),
                ('SqueezeSliceData', dict(kind='data')),
            ],
            edges=[
                ('EnterMaxIndex', 'EnterMaxIndexData'),
                ('EnterMaxIndexData', 'Less', {'in': 1}),
                ('Less', 'LessData'),
                ('LessData', 'LoopCond'),
                ('LoopCond', 'LoopCondData'),
                ('LoopCondData', 'SwitchInput', {'in': 1}),

                ('EnterInput', 'EnterInputData'),
                ('EnterInputData', 'MergeInput', {'in': 0}),
                ('MergeInput', 'MergeInputData'),
                ('MergeInputData', 'SwitchInput', {'in': 0}),
                ('SwitchInput', 'SwitchInputData', {'out': 1}),
                ('SwitchInputData', 'IdentityInput'),
                ('IdentityInput', 'IdentityInputData'),
                ('IdentityInputData', 'Gather', {'in': 0}),
                ('IdentityInputData', 'NextIterationInput'),

                ('InitIndex', 'InitIndexData'),
                ('InitIndexData', 'EnterIndex'),
                ('EnterIndex', 'EnterIndexData'),
                ('EnterIndexData', 'MergeIndex', {'in': 0}),
                ('MergeIndex', 'MergeIndexData'),
                ('MergeIndexData', 'SwitchIndex', {'in': 0}),
                ('MergeIndexData', 'Less', {'in': 0}),
                ('LoopCondData', 'SwitchIndex', {'in': 1}),
                ('SwitchIndex', 'SwitchIndexData', {'out': 1}),
                ('SwitchIndexData', 'IdentityIndex'),
                ('IdentityIndex', 'IdentityIndexData'),
                ('IdentityIndexData', 'AddIndex', {'in': 0}),
                ('AddIndex', 'AddIndexData'),
                ('AddIndexData', 'NextIterationIndex'),
                ('IndexDelta', 'IndexDeltaData'),
                ('IndexDeltaData', 'AddIndex', {'in': 1}),

                ('IdentityIndexData', 'UnsqueezeIndex'),
                ('UnsqueezeIndex', 'UnsqueezeIndexData'),
                ('UnsqueezeIndexData', 'Gather', {'in': 1}),
                ('Axis', 'AxisData'),
                ('AxisData', 'Gather', {'in': 2}),
                ('Gather', 'GatherData'),
                ('GatherData', 'SqueezeSlice'),
                ('SqueezeSlice', 'SqueezeSliceData'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        # retrieve attribute values for TensorIteratorInput node
        init_time = match['InitIndex'].value.item(0)
        time_step = match['IndexDelta'].value.item(0)
        axis = match['Axis'].value.item(0)

        # retrieve input and output nodes for TensorIteratorInput node
        initial_input_node = match['EnterInput']
        current_index_node = match['IdentityIndex']
        size_node = match['EnterMaxIndex']
        resulted_slice_node = match['SqueezeSlice']
        resulted_slice_node_name = resulted_slice_node.soft_get('name', resulted_slice_node.id)

        # create TensorIteratorInput node that reflects slicing of input for each time step along axis
        ti_input_node = TensorIteratorInput(graph, dict(axis=axis, start=init_time, stride=time_step,
                                                        name=resulted_slice_node_name + '/TensorIteratorInput')
                                            ).create_node()
        size_node.in_port(0).get_connection().add_destination(ti_input_node.in_port(0))
        initial_input_node.in_port(0).get_connection().set_destination(ti_input_node.in_port(1))
        current_index_node.out_port(0).connect(ti_input_node.in_port(2))
        resulted_slice_node.out_port(0).get_connection().set_source(ti_input_node.out_port(0))

        # delete no longer needed nodes responsible for slicing of input in the original graph
        node_names_for_remove = ['EnterInput', 'MergeInput', 'SwitchInput',
                                 'IdentityInput', 'NextIterationInput', 'SqueezeSlice', 'UnsqueezeIndex', 'Gather']
        graph.remove_nodes_from([match[node_name].id for node_name in node_names_for_remove])
