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
import logging as log

import numpy as np

from extensions.middle.TensorIteratorCondition import looking_for_op_in_list
from extensions.ops.elementwise import Mul
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class GNMT_sequence_lengths(FrontReplacementPattern):
    """
    This pass matching GNMT-like condition (like in DynamicDecoderConditionMatcher) with path for sequence lengths
    calculation:
        Seq_len_data -> Max -> Cast -> Mul -> Round -> Cast.

    After matching this pattern:
        1. This replacer looking for encoder sequence lengths node (using information about encoder condition stucture)
        2. Create node for multiplying Encoder sequence lengths by 2 (as it works in GNMT).
        3. Connect Encoder sequence lengths value multiplied by 2 with decoder TensorArrays as size.
    """
    enabled = True

    @staticmethod
    def pattern():
        log.debug('+++++++++++++++ GNMT Sequence Lengths ConditionMatching ++++++++++++++++')
        return dict(
            nodes=[
                ('loop_cond', dict(kind='op', op='LoopCond')),
                ('logical_not', dict(kind='op', op='Not')),

                ('all', dict(kind='op', op='ReduceAnd')),

                ('Merge_16', dict(kind='op', op='Merge')),

                ('NextIteration_16', dict(kind='op', op='NextIteration')),

                ('Switch', dict(kind='op', op='Switch')),

                ('Identity', dict(kind='op', op='Identity')),

                ('Switch_1', dict(kind='op', op='Switch')),

                ('Identity_1', dict(kind='op', op='Identity')),

                ('add', dict(kind='op', op='Add')),

                ('Less_enter',  dict(kind='op', op='Enter')),

                ('And', dict(kind='op', op='LogicalAnd')),

                ('Less',  dict(kind='op', op='Less')),
                ('TensorArrayWrite', dict(kind='op', op='TensorArrayWriteV3')),
                ('TensorArrayWrite_1', dict(kind='op', op='TensorArrayWriteV3')),

                ('Max', dict(kind='op', op='ReduceMax')),
                ('ToFloat', dict(kind='op', op='Cast')),
                ('Mul', dict(kind='op', op='Mul')),
                ('Round', dict(kind='op', op='Round')),
                ('ToInt', dict(kind='op', op='Cast')),
            ],
            edges=[
                ('NextIteration_16', 'Merge_16'),
                ('Merge_16', 'all'),

                ('all', 'logical_not'),

                ('Less_enter','Less'),

                ('Less', 'And'),

                ('logical_not', 'And'),
                ('And', 'loop_cond'),

                ('loop_cond', 'Switch'),
                ('Switch', 'Identity'),
                ('Identity', 'add'),

                ('loop_cond', 'Switch_1'),
                ('Switch_1', 'Identity_1'),

                ('Identity_1', 'TensorArrayWrite'),
                ('Identity_1', 'TensorArrayWrite_1'),

                ('Max', 'ToFloat'),
                ('ToFloat', 'Mul'),
                ('Mul', 'Round'),
                ('Round', 'ToInt'),
                ('ToInt', 'Less_enter'),
            ],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        log.debug('================== GNMTBeforeConditionFind ==================')
        input_sequence_lengths = match['Max'].in_port(0).get_source().node
        encoder_sequence_lengths = looking_for_op_in_list([port.node for port in input_sequence_lengths.out_port(0).get_destinations()],
                                                          'Identity')

        # Looking for Sequence_length node in encoder looks like:
        # Sequence_length -> CheckSeqLen -> Max -> Maximum -> Minimum

        check_seq_len = looking_for_op_in_list([port.node for port in encoder_sequence_lengths.out_port(0).get_destinations()],
                                               'Identity')
        max = looking_for_op_in_list([port.node for port in check_seq_len.out_port(0).get_destinations()], 'ReduceMax')
        maximum = max.out_port(0).get_destinations()[0].node
        assert maximum.op == 'Maximum'
        minimum = maximum.out_port(0).get_destinations()[0].node
        assert minimum.op == 'Minimum'

        tensor_seq_len = looking_for_op_in_list([minimum.in_port(port).get_source().node for port in minimum.in_ports()], 'StridedSlice')

        # Create node for multiplying seq_len by 2
        const = Const(graph, {'name': 'FakeSeqLenMultiplyer', 'value': np.array(2)}).create_node()
        mul_op = Mul(graph, {'name': 'FakeSeqLen'}).create_node()

        const.out_port(0).get_connection().set_destination(mul_op.in_port(1))
        tensor_seq_len.out_port(0).get_connection().add_destination(mul_op.in_port(0))

        # Connect seq_len * 2 to TensorArray from GNMT loop
        ta_writes = [port.node for port in match['Identity_1'].out_port(0).get_destinations() if port.node.op == 'TensorArrayWriteV3']

        for ta_write in ta_writes:
            ta = ta_write.in_port(0).get_source().node.in_port(0).get_source().node

            ta.in_port(0).disconnect()
            ta.in_port(0).get_connection().set_source(mul_op.out_port(0))
