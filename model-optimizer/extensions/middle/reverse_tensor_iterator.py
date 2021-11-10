# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
from extensions.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
from mo.front.common.partial_infer.utils import is_fully_defined
from mo.graph.graph import Graph, Node
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


def update_ti(ti, direct_reverse):
    seq_axis = direct_reverse.seq_axis
    # Modify stride in TI
    for port_map in [ti.input_port_map, ti.output_port_map]:
        for port in port_map:
            if 'axis' in port and port['axis'] is not None and 'external_port_id' in port:
                assert port['axis'] == seq_axis, \
                    'axis == {} != {} == direct_reverse.seq_dim'.format(port['axis'], seq_axis)
                if 'stride' not in port or port['stride'] is None:
                    port['stride'] = 1
                assert port['stride'] in [-1, 1]
                port['stride'] = -port['stride']
                if port['stride'] == -1:
                    port['start'] = -1
                    port['end'] = 0
                elif port['stride'] == 1:
                    port['start'] = None
                    port['end'] = None


class ReverseTensorIteratorLSTM(MiddleReplacementPattern):
    """ Fuses Reverse operations around TI: ReverseSequence --> TI  --> ReverseSequence.

        WARNING This transformation is limited to support of very special case of TI but
        code doesn't check all the cases.
    """

    enabled = True

    def run_after(self):
        return [
            ONNXRNNSequenceNormalize,
            TransposeTensorIteratorLSTM,
        ]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @staticmethod
    def is_fusable_reverse_sequence(node: Node):
        sequence_lengths = node.in_port(1).data.get_value()
        assert sequence_lengths is not None
        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        seq_len = input_shape[node.seq_axis]
        if is_fully_defined(sequence_lengths):
            return np.all(sequence_lengths == seq_len)
        else:
            # check that we take sequence_length from input shape based on ReverseV2ToReverseSequence transformation
            broadcast_node = node.in_port(1).get_source().node
            if broadcast_node.op != 'Broadcast':
                return False
            gather_node = broadcast_node.in_port(0).get_source().node
            if gather_node.op != "Gather" or \
                    (np.all(gather_node.in_port(1).data.get_value() != [0]) or
                     np.all(gather_node.in_port(2).data.get_value() != [node.seq_axis])):
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

                ('const', dict(type='Const')),
                ('const_d', dict(kind='data')),

                ('direct_reverse', dict(op='ReverseSequence')),
                ('input_reversed', dict(kind='data')),
                ('init_hidden', dict(kind='data')),

                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_reversed', dict(kind='data')),

                ('const_1', dict(type='Const')),
                ('const_1_d', dict(kind='data')),

                ('inverse_reverse', dict(op='ReverseSequence')),
                ('output', dict(kind='data')),
            ],
            edges=[
                ('input', 'direct_reverse', {'in': 0}),
                ('const', 'const_d'),
                ('const_d', 'direct_reverse', {'in': 1}),
                ('direct_reverse', 'input_reversed'),

                ('input_reversed', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('ti', 'output_reversed', {'out': 0}),

                ('output_reversed', 'inverse_reverse', {'in': 0}),
                ('const_1', 'const_1_d'),
                ('const_1_d', 'inverse_reverse', {'in': 1}),
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

        update_ti(ti, direct_reverse)

        # Remove reverses
        remove_op_node_with_data_node(graph, direct_reverse)
        remove_op_node_with_data_node(graph, inverse_reverse)


class ReverseTensorIteratorLSTMWithSqueeze(MiddleReplacementPattern):
    """ Fuses Reverse operations around TI: ReverseSequence --> Squeeze --> TI --> UnSqueeze --> ReverseSequence.

        WARNING This transformation is limited to support only case described before but code doesn't check all
        the cases.
    """

    enabled = True
    force_clean_up = True

    def run_after(self):
        return [
            ONNXRNNSequenceNormalize,
            TransposeTensorIteratorLSTM,
        ]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('input_2', dict(kind='data')),
                ('input_3', dict(kind='data')),

                ('unsqueeze', dict(op='Unsqueeze')),
                ('unsqueeze_d', dict(kind='data')),
                ('direct_reverse', dict(op='ReverseSequence')),
                ('input_reversed', dict(kind='data')),
                ('squeeze', dict(op='Squeeze')),
                ('squeeze_d', dict(kind='data')),

                ('init_hidden', dict(kind='data')),

                ('ti', dict(kind='op', op='TensorIterator')),

                ('output_reversed', dict(kind='data')),

                ('unsqueeze_1', dict(op='Unsqueeze')),
                ('unsqueeze_1_d', dict(kind='data')),
                ('inverse_reverse', dict(op='ReverseSequence')),
                ('output', dict(kind='data')),
                ('squeeze_1', dict(op='Squeeze')),
                ('squeeze_1_d', dict(kind='data')),
            ],
            edges=[
                ('input', 'unsqueeze'),
                ('unsqueeze', 'unsqueeze_d'),
                ('unsqueeze_d', 'direct_reverse', {'in': 0}),
                ('input_2', 'direct_reverse', {'in': 1}),
                ('direct_reverse', 'input_reversed'),
                ('input_reversed', 'squeeze'),
                ('squeeze', 'squeeze_d'),

                ('squeeze_d', 'ti', {'in': 0}),
                ('init_hidden', 'ti', {'in': 1}),
                ('ti', 'output_reversed', {'out': 0}),

                ('output_reversed', 'unsqueeze_1'),
                ('unsqueeze_1', 'unsqueeze_1_d'),
                ('unsqueeze_1_d', 'inverse_reverse', {'in': 0}),
                ('input_3', 'inverse_reverse', {'in': 1}),
                ('inverse_reverse', 'output'),
                ('output', 'squeeze_1'),
                ('squeeze_1', 'squeeze_1_d'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        ti = match['ti']
        direct_reverse = match['direct_reverse']
        inverse_reverse = match['inverse_reverse']
        squeeze = match['squeeze']
        unsqueeze = match['unsqueeze']
        squeeze_1 = match['squeeze_1']
        unsqueeze_1 = match['unsqueeze_1']

        assert direct_reverse.seq_axis == inverse_reverse.seq_axis
        assert direct_reverse.batch_axis is None and inverse_reverse.batch_axis is None or \
               direct_reverse.batch_axis == inverse_reverse.batch_axis

        if not ReverseTensorIteratorLSTM.is_fusable_reverse_sequence(direct_reverse) or \
                not ReverseTensorIteratorLSTM.is_fusable_reverse_sequence(inverse_reverse):
            # we can not merge ReverseSequence without equal sequences
            return

        direct_reverse.seq_axis = direct_reverse.seq_axis - 1 if direct_reverse.seq_axis > 0 else direct_reverse.seq_axis
        update_ti(ti, direct_reverse)

        # Remove reverses
        unsqueeze.out_port(0).get_destinations()
        in_unsqueeze = unsqueeze.in_port(0).get_source()
        for dest in squeeze.out_port(0).get_destinations():
            dest.get_connection().set_source(in_unsqueeze)
        unsqueeze.in_port(0).disconnect()

        in_unsqueeze_1 = unsqueeze_1.in_port(0).get_source()
        for dest in squeeze_1.out_port(0).get_destinations():
            dest.get_connection().set_source(in_unsqueeze_1)
        unsqueeze_1.in_port(0).disconnect()
        graph.remove_node(direct_reverse.id)
        graph.remove_node(inverse_reverse.id)
