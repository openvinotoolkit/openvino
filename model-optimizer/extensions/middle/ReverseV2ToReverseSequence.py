# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.middle.InsertLayoutPropagationTransposes import mark_output_as_in_correct_layout, \
    mark_input_as_in_correct_layout
from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.broadcast import Broadcast
from mo.ops.shape import Shape
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.shape import node_to_get_shape_value_of_indices


class ReverseToReverseSequence(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.PartialInfer import PartialInfer
        return [PartialInfer]

    def run_before(self):
        from extensions.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM, \
            ReverseTensorIteratorLSTMWithSqueeze
        return [ReverseTensorIteratorLSTM, ReverseTensorIteratorLSTMWithSqueeze]

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
        reverse_name = reverse.soft_get('name', reverse.id)

        assert reverse.in_port(1).disconnected()

        # add new dimension as batch if rank = 1
        unsq_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                     {'name': reverse_name+"/Unsqueeze"})
        reverse.in_port(0).get_source().connect(unsq_node.in_port(0))
        # set layout as correct to avoid adding Transpose nodes
        unsq_node.in_port(1).__setattr__('input_permutation', None)
        mark_input_as_in_correct_layout(unsq_node, 0)
        mark_output_as_in_correct_layout(unsq_node, 0)
        new_in = unsq_node.out_port(0)
        batch_axis = int64_array([0])
        seq_axis = reverse['axis'] + 1 if reverse['axis'] >= 0 else reverse['axis']  # add 1 for newly added dimension

        # 1. For ReverseSequence 1-port input is seq_lengths => create this input node
        reverse_name = reverse.soft_get('name',  reverse.id)
        rename_node(reverse, reverse_name + '/to_delete')

        shape_node = Shape(graph, {'name': reverse_name + "/shape"}).create_node()
        new_in.connect(shape_node.in_port(0))
        mark_input_as_in_correct_layout(shape_node, 0)
        seq_axis_node = node_to_get_shape_value_of_indices(shape_node, [seq_axis])
        mark_input_as_in_correct_layout(seq_axis_node, 0)
        batch_node = node_to_get_shape_value_of_indices(shape_node, batch_axis)
        mark_input_as_in_correct_layout(batch_node, 0)
        broadcast_node = Broadcast(graph, {'name': reverse_name + "/broadcast"}).create_node()
        broadcast_node.in_port(0).connect(seq_axis_node.out_port(0))
        broadcast_node.in_port(1).connect(batch_node.out_port(0))
        mark_input_as_in_correct_layout(broadcast_node, 0)
        mark_input_as_in_correct_layout(broadcast_node, 1)

        # 2. Create new ReverseSequence node and reconnect all inputs/outputs to it
        reverse_sequence = ReverseSequence(graph, {'name':  reverse_name, 'seq_axis': seq_axis,
                                                   'batch_axis': batch_axis}).create_node()
        reverse_sequence.in_port(0).connect(new_in)
        reverse_sequence.in_port(1).connect(broadcast_node.out_port(0))
        mark_input_as_in_correct_layout(reverse_sequence, 0)
        mark_input_as_in_correct_layout(reverse_sequence, 1)

        # remove added dimension
        squeeze_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                        {'name': reverse_name + "/Squeeze"})
        squeeze_node.in_port(0).connect(reverse_sequence.out_port(0))
        mark_input_as_in_correct_layout(squeeze_node, 0)

        reverse.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
        mark_output_as_in_correct_layout(squeeze_node, 0)

        # 3. Delete old Reverse node
        graph.remove_node(reverse.id)
