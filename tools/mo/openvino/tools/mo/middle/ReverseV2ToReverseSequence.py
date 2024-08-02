# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.reverse_sequence import ReverseSequence
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices


class ReverseToReverseSequence(MiddleReplacementPattern):
    """
    Transformation converts Reverse to ReverseSequence operation.
    Parameters for ReverseSequence calculates in the following way:
     * seq_axis - set axis value from Reverse operation
     * batch_axis - set 0 if seq_axis is not 0 otherwise set 1
     * seq_lengths - take from shape shape[seq_axis] value and broadcast it to vector with shape[batch_axis] length
    If input is 1D tensor then we add one more dimension to set different seq_axis and batch_axis.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.PartialInfer import PartialInfer
        return [PartialInfer]

    def run_before(self):
        from openvino.tools.mo.middle.reverse_tensor_iterator import ReverseTensorIteratorLSTM
        return [ReverseTensorIteratorLSTM]

    def find_and_replace_pattern(self, graph: Graph):
        reverse_nodes = graph.get_op_nodes(op='Reverse')
        for reverse in reverse_nodes:
            reverse_name = reverse.soft_get('name', reverse.id)

            assert reverse.in_port(1).disconnected()
            assert reverse.has_valid('axis')

            in_shape_rank = len(reverse.in_port(0).data.get_shape())
            # 1. Add new dimension as batch for rank = 1 to have batch != seq_axis
            if in_shape_rank == 1:
                unsq_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([0]),
                                                             {'name': reverse_name+"/Unsqueeze"})
                reverse.in_port(0).get_source().connect(unsq_node.in_port(0))
                new_in = unsq_node.out_port(0)
                batch_axis = 0
                seq_axis = 1
            else:
                new_in = reverse.in_port(0).get_source()
                seq_axis = reverse['axis']
                batch_axis = 0 if seq_axis != 0 else 1

            # 2. For ReverseSequence 1-port input is seq_lengths => create this input node as
            # shape[seq_axis] broadcasted to shape[batch_axis]
            # in ---> ShapeOf ----> Gather(seq_axis)  ----> Broadcast----->
            #            |                                      |
            #            | -------> Gather(batch_axis)----------|
            shape_node = Shape(graph, {'name': reverse_name + "/Shape"}).create_node()
            new_in.connect(shape_node.in_port(0))
            seq_axis_node = node_to_get_shape_value_of_indices(shape_node, [seq_axis])
            batch_node = node_to_get_shape_value_of_indices(shape_node, [batch_axis])
            broadcast_node = Broadcast(graph, {'name': reverse_name + "/Broadcast"}).create_node()
            broadcast_node.in_port(0).connect(seq_axis_node.out_port(0))
            broadcast_node.in_port(1).connect(batch_node.out_port(0))

            # 3. Create new ReverseSequence node and reconnect all inputs/outputs to it
            rename_node(reverse, reverse_name + '/to_delete')
            reverse_sequence = ReverseSequence(graph, {'name':  reverse_name, 'seq_axis': seq_axis,
                                                       'batch_axis': batch_axis}).create_node()
            reverse_sequence.in_port(0).connect(new_in)
            reverse_sequence.in_port(1).connect(broadcast_node.out_port(0))

            # 4. remove added dimension for rank = 1
            if in_shape_rank == 1:
                rename_node(reverse_sequence, reverse_name + '/ReverseSequence')
                squeeze_node = create_op_node_with_second_input(graph, Squeeze, int64_array([0]),
                                                                {'name': reverse_name})
                squeeze_node.in_port(0).connect(reverse_sequence.out_port(0))
                reverse.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
            else:
                reverse.out_port(0).get_connection().set_source(reverse_sequence.out_port(0))

        # 5. Delete old Reverse node
        graph.remove_nodes_from([reverse.id for reverse in reverse_nodes])
