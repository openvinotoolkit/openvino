# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
from openvino.tools.mo.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ReverseTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Reverse operations around TI: ReverseSequence --> TI  --> ReverseSequence.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True
    force_clean_up = True

    def run_after(self):
        return [
            ONNXRNNSequenceNormalize,
            TransposeTensorIteratorLSTM,
        ]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @staticmethod
    def is_fusable_reverse_sequence(node: Node):
        sequence_lengths = node.in_port(1).data.get_value()
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        seq_len = input_shape[node.seq_axis]
        if sequence_lengths is not None and is_fully_defined(sequence_lengths) and is_fully_defined(seq_len):
            return np.all(sequence_lengths == seq_len)
        else:
            # check that we take sequence_length from input shape based on ReverseV2ToReverseSequence transformation
            broadcast_node = node.in_port(1).get_source().node
            if broadcast_node.op != 'Broadcast':
                return False
            gather_node = broadcast_node.in_port(0).get_source().node
            if gather_node.op != "Gather" or \
                    (np.all(gather_node.in_port(2).data.get_value() != [0]) or
                     np.all(gather_node.in_port(1).data.get_value() != [node.seq_axis])):
                return False
            gather_node_2 = broadcast_node.in_port(1).get_source().node
            if gather_node_2.op != "Gather" or \
                    (np.all(gather_node_2.in_port(2).data.get_value() != [0]) or
                     np.all(gather_node_2.in_port(1).data.get_value() != [node.batch_axis])):
                return False
            shape_node = gather_node.in_port(0).get_source().node
            if shape_node.op != "ShapeOf":
                return False
            if shape_node.in_port(0).get_source().node != node.in_port(0).get_source().node:
                return False

            return True

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),

                ('direct_seq_len_d', dict(kind='data')),
                ('direct_reverse', dict(op='ReverseSequence')),
                ('input_reversed', dict(kind='data')),
                ('init_hidden', dict(kind='data')),

                ('ti', dict(kind='op', op='TensorIterator')),
                ('output_reversed', dict(kind='data')),

                ('inverse_seq_len_d', dict(kind='data')),
                ('inverse_reverse', dict(op='ReverseSequence')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'direct_reverse', {'in': 0}),
                ('direct_seq_len_d', 'direct_reverse', {'in': 1}),
                ('direct_reverse', 'input_reversed'),

                ('input_reversed', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('ti', 'output_reversed', {'out': 0}),

                ('output_reversed', 'inverse_reverse', {'in': 0}),
                ('inverse_seq_len_d', 'inverse_reverse', {'in': 1}),
                ('inverse_reverse', 'output'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        ti = match['ti']
        direct_reverse = match['direct_reverse']
        inverse_reverse = match['inverse_reverse']

        assert direct_reverse.seq_axis == inverse_reverse.seq_axis
        assert direct_reverse.batch_axis is None and inverse_reverse.batch_axis is None or \
               direct_reverse.batch_axis == inverse_reverse.batch_axis

        if not self.is_fusable_reverse_sequence(direct_reverse) or \
                not self.is_fusable_reverse_sequence(inverse_reverse):
            # we can not merge ReverseSequence without equal sequences
            return

        # Modify stride in TI
        for port_map in [ti.input_port_map, ti.output_port_map]:
            for port in port_map:
                if 'axis' in port and port['axis'] is not None and 'external_port_id' in port:
                    assert port['axis'] == direct_reverse.seq_axis, \
                        'axis == {} != {} == direct_reverse.seq_dim'.format(port['axis'], direct_reverse.seq_axis)
                    if 'stride' not in port or port['stride'] is None:
                        port['stride'] = 1
                    assert port['stride'] in [-1, 1]
                    port['stride'] = -port['stride']
                    if port['stride'] == -1:
                        port['start'] = -1
                        port['end'] = 0
                    elif port['stride'] == 1:
                        port['start'] = 0
                        port['end'] = -1

        # disconnect subgraph for seq length calculation
        direct_reverse.in_port(1).disconnect()
        inverse_reverse.in_port(1).disconnect()
        # Remove reverses
        remove_op_node_with_data_node(graph, direct_reverse)
        remove_op_node_with_data_node(graph, inverse_reverse)
