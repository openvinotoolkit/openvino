# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern
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
        reverse_name = reverse.soft_get('name', reverse.id)

        assert reverse.in_port(1).disconnected()

        # add new dimension as batch
        unsq_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                     {'name': reverse_name+"/Unsqueeze"})
        reverse.in_port(0).get_connection().set_destination(unsq_node.in_port(0))
        batch_axis = int64_array([0])
        seq_axis = reverse['axis'] + 1  # add 1 for newly added dimension

        # 1. For ReverseSequence 1-port input is seq_lengths => create this input node
        reverse_name = reverse.soft_get('name',  reverse.id)
        rename_node(reverse, reverse_name + '/to_delete')

        shape_node = Shape(graph, {'name': reverse_name + "/shape"}).create_node()
        unsq_node.out_port(0).connect(shape_node.in_port(0))
        seq_axis_node = node_to_get_shape_value_of_indices(shape_node, [seq_axis])

        # 2. Create new ReverseSequence node and reconnect all inputs/outputs to it
        reverse_sequence = ReverseSequence(graph, {'name':  reverse_name, 'seq_axis': seq_axis,
                                                   'batch_axis': batch_axis}).create_node()
        reverse_sequence.in_port(0).connect(unsq_node.out_port(0))
        reverse_sequence.in_port(1).connect(seq_axis_node.out_port(0))

        # remove added dimension
        squeeze_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                        {'name': reverse_name + "/Squeeze"})
        squeeze_node.in_port(0).connect(reverse_sequence.out_port(0))

        reverse.out_port(0).get_connection().set_source(squeeze_node.out_port(0))

        # 3. Delete old Reverse node
        graph.remove_node(reverse.id)
