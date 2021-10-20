# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Mul
from extensions.ops.reverse_sequence import ReverseSequence
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const
from mo.ops.shape import Shape
from mo.utils.error import Error
from mo.utils.shape import node_to_get_shape_value_of_indices


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
        reverse_name = reverse.soft_get('name',  reverse.id)
        rename_node(reverse, reverse_name + '/to_delete')

        one_const = Const(graph, {'name': reverse_name + "/one",
                                  'value': np.ones([1])}).create_node()
        shape_node = Shape(graph, {'name': reverse_name + "/shape"}).create_node()
        reverse.in_port(0).get_source().connect(shape_node.in_port(0))
        batch_node = node_to_get_shape_value_of_indices(shape_node, [batch_axis])
        seq_axis_node = node_to_get_shape_value_of_indices(shape_node, [seq_axis])
        seq_length_shape_node = Mul(graph, {'name': reverse_name + "/mul"}).create_node()
        seq_length_shape_node.in_port(0).connect(batch_node.out_port(0))
        seq_length_shape_node.in_port(1).connect(seq_axis_node.out_port(0))
        broadcast_node = Broadcast(graph, {'name': reverse_name + "/broadcast"}).create_node()
        broadcast_node.in_port(0).connect(one_const.out_port(0))
        broadcast_node.in_port(1).connect(seq_length_shape_node.out_port(0))

        # 2. Create new ReverseSequence node and reconnect all inputs/outputs to it
        reverse_sequence = ReverseSequence(graph, {'name':  reverse_name, 'seq_axis': seq_axis,
                                                   'batch_axis': batch_axis}).create_node()
        reverse_sequence.in_port(1).connect(broadcast_node.out_port(0))

        rename_node(reverse_sequence, reverse_name)
        reverse.in_port(0).get_connection().set_destination(reverse_sequence.in_port(0))
        reverse.out_port(0).get_connection().set_source(reverse_sequence.out_port(0))

        # 3. Delete old Reverse node
        graph.remove_node(reverse.id)
