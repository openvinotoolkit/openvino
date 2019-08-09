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

from extensions.ops.reverse_sequence import ReverseSequence
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.utils.error import Error


class ReverseToReverseSequence(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.PartialInfer import PartialInfer
        return [PartialInfer]

    def run_before(self):
        from extensions.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM
        return [ReverseTensorIteratorLSTM]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('reverse', dict(kind='op', op='Reverse'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        reverse = match['reverse']
        input_data_shape = reverse.in_node(0).shape

        if len(input_data_shape) == 1:
            raise Error('Reverse operation name = {} is\'t supported because of 1D input.'.format(reverse.name))

        assert reverse.in_port(1).disconnected()

        seq_axis = reverse['axis']
        # We need to choose arbitrary batch_axis != sequence_axis
        batch_axis = int(not seq_axis)

        # 1. For ReverseSequence 1-port input is seq_lengths => create this input node
        seq_lengths = np.ones(input_data_shape[batch_axis]) * input_data_shape[seq_axis]
        const = Const(graph, dict(value=seq_lengths)).create_node()

        # 2. Create new ReverseSequence node and reconnect all inputs/outputs to it
        reverse_sequence = ReverseSequence(graph, {'name': reverse.name + '/ReverseSequence/',
                                                   'seq_axis': seq_axis, 'batch_axis': batch_axis}).create_node()

        reverse.in_port(0).get_connection().set_destination(reverse_sequence.in_port(0))
        const.out_port(0).connect(reverse_sequence.in_port(1))
        reverse.out_port(0).get_connection().set_source(reverse_sequence.out_port(0))

        # 3. Delete old Reverse node
        graph.remove_node(reverse.id)
