# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.utils.error import Error


class SequenceLengthToMask(MiddleReplacementPattern):
    """
    Convert a sequence length to a sequence mask for CTCGreedyDecoder if its value is available.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def find_and_replace_pattern(self, graph: Graph):
        for ctc_greedy_decoder in graph.get_op_nodes(op='CTCGreedyDecoder', use_mask_format=True):
            ctc_greedy_decoder_name = ctc_greedy_decoder.soft_get('name', ctc_greedy_decoder.id)

            sequence_length_value = ctc_greedy_decoder.in_port(1).data.get_value()
            if sequence_length_value is None:
                raise Error('The second input to the CTCGreedyDecoder node "{}" is not constant. This case is not '
                            'supported with the Inference Engine.'.format(ctc_greedy_decoder_name))

            # transform a sequence length to a sequence mask
            logits_shape = ctc_greedy_decoder.in_port(0).data.get_shape()
            assert logits_shape is not None and len(logits_shape) == 3, \
                "Incorrect shape for logits input of {} node".format(ctc_greedy_decoder_name)
            batch_size = logits_shape[1]
            time_size = logits_shape[0]
            mask_value = np.zeros([batch_size, time_size], dtype=np.float)
            for sample_ind, sample_seq_length in enumerate(sequence_length_value):
                mask_value[sample_ind, 0:sample_seq_length] = 1
            mask_value = np.transpose(mask_value)

            # create Const node with computed mask value
            mask_node = Const(graph, {'name': ctc_greedy_decoder_name + '/Mask',
                                      'value': mask_value}).create_node()

            # connect computed mask to CTCGreedyDecoder node
            ctc_greedy_decoder.in_port(1).get_connection().set_source(mask_node.out_port(0))

            # remove attribute-marker
            del ctc_greedy_decoder['use_mask_format']
