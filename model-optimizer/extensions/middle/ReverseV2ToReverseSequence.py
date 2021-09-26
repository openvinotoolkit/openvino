# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern
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

        reverse_name = reverse.soft_get('name',  reverse.id)
        rename_node(reverse, reverse_name + '/to_delete')
        # 2. Create new ReverseSequence node and reconnect all inputs/outputs to it
        reverse_sequence = create_op_node_with_second_input(graph, ReverseSequence, seq_lengths,
                                                            {'name':  reverse_name, 'seq_axis': seq_axis,
                                                             'batch_axis': batch_axis})
        rename_node(reverse_sequence, reverse_name)
        reverse.in_port(0).get_connection().set_destination(reverse_sequence.in_port(0))
        reverse.out_port(0).get_connection().set_source(reverse_sequence.out_port(0))

        # 3. Delete old Reverse node
        graph.remove_node(reverse.id)
