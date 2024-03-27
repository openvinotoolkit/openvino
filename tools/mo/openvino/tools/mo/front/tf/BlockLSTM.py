# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error


class BlockLSTM(FrontReplacementPattern):
    """
    We prepare TensorFlow BlockLSTM op to be replaced with LSTMSequence op that will be repacked to TensorIterator later

    TensorFlow BlockLSTM op description:

        Op parameters:
         cell_clip:    Value to clip the 'cs' value to.
         use_peephole: Whether to use peephole weights.
         forget_bias:  The forget gate bias.

        Inputs:
         0: seq_len_max:  Maximum time length actually used by this input. Outputs are padded with 0s beyond this length
         1: x:            The sequence input to the LSTM, shape (timelen, batch_size, num_inputs)
         2: cs_prev:      Value of the initial cell state
         3: h_prev:       Initial output of cell (to be used for peephole)
         4: w:            The weight matrix
         5: wci:          The weight matrix for input gate peephole connection
         6: wcf:          The weight matrix for forget gate peephole connection
         7: wco:          The weight matrix for output gate peephole connection
         8: b:            The bias vector

        Outputs:
         0: i:            The input gate                    over the whole time sequence
         1: cs:           The cell state before the tanh    over the whole time sequence
         2: f:            The forget gate                   over the whole time sequence
         3: o:            The output gate                   over the whole time sequence
         4: ci:           The cell input                    over the whole time sequence
         5: co:           The cell after the tanh           over the whole time sequence
         6: h:            The output h vector               over the whole time sequence

    Limitations:
    - peephole connection, so we check `use_peephole`!=True and cut `wci`, `wco`, `wcf` off
    - cell_clip parameter, so we check `cell_clip==-1`, which means we do not clip
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='BlockLSTM'):
            if node.use_peephole:
                raise Error("BlockLSTM operation is not supported with `use_peephole`==True. Node: {}"
                            "".format(node.soft_get('name')))

            if node.cell_clip != -1:
                raise Error("Clipping is not supported for BlockLSTM operation. `cell_clip`={!s} for node: {}"
                            "".format(node.cell_clip, node.soft_get('name')))

            log.debug("Start BlockLSTM->LSTMSequence translation for node: {} with parameters:\n"
                      "`cell_clip`={!s}, `use_peephole`=={!s}, `forget_bias`={!s}\n"
                      "inputs: {},\noutputs:{}".format(node.soft_get('name'), node.cell_clip, node.use_peephole,
                                                       node.forget_bias, {p: i.id for p, i in node.in_nodes().items()},
                                                       {p: o.id for p, o in node.out_nodes().items()}))

            log.debug("Cutting all inputs for peephole connection (5, 6, 7 input ports) off, as `use_peephole`=False")
            log.debug("Cutting seq_len_max input off")

            # disconnect all peephole releated inputs and seq_len_max
            for port_idx in [0, 5, 6, 7]:
                if node.is_in_port_connected(port_idx):
                    node.in_port(port_idx).disconnect()

            assert node.is_in_port_connected(1), "Sequence input to the BlockLSTM is required (1 port). Node {}".format(
                node.id)
            assert node.is_in_port_connected(2), "Value of the initial cell state is required (2 port). Node {}".format(
                node.id)
            assert node.is_in_port_connected(
                3), "Initial output of cell is required input to BlockLSTM (3 port). Node {}".format(node.id)
            assert node.is_in_port_connected(
                4), "The weight matrix is required input to BlockLSTM (4 port) . Node {}".format(node.id)
            assert node.is_in_port_connected(
                8), "The bias vector is required input to BlockLSTM (8 port). Node {}".format(node.id)

            # reconnect inputs since OpenVINO LSTMSequence requires different order
            # Reconnecting input edges of LSTMSequence:
            # TF input edges:             Description:                 MO input edges:
            #       1                          input                        0
            #       4                         weights                       1
            #       8                         biases                        2
            #       3               h_prev: initial output of cell          3
            #       2               cs_prev: initial cell state             4

            input_source = node.in_port(1).get_source()
            weights_source = node.in_port(4).get_source()
            biases_source = node.in_port(8).get_source()
            h_prev_source = node.in_port(3).get_source()
            cs_prev_source = node.in_port(2).get_source()

            node.in_port(0).get_connection().set_source(input_source)
            node.in_port(1).get_connection().set_source(weights_source)
            node.in_port(2).get_connection().set_source(biases_source)
            node.in_port(3).get_connection().set_source(h_prev_source)
            node.in_port(4).get_connection().set_source(cs_prev_source)
            # disconnect original bias input that is no longer needed
            if node.is_in_port_connected(8):
                node.in_port(8).disconnect()

            # check that all outputs unsupported by OpenVINO LSTMSequence are absent
            for output_port_idx in [0, 2, 3, 4, 5]:
                if node.is_out_port_connected(output_port_idx):
                    raise Error("Output port {} of BlockLSTM node {} is not supported".format(node.id, output_port_idx))

            # Reconnecting output edges of LSTMSequence:
            # TF output edges:             Description:                 MO output edges:
            #       6                     output h vector                     0
            #       1                   cell state before the tanh            1

            # we need to move only 6-th output port to 0-th port
            if node.is_out_port_connected(6):
                node.add_output_port(0, skip_if_exist=True)
                node.out_port(6).get_connection().set_source(node.out_port(0))
                node.out_port(6).disconnect()
                node.delete_output_port(6, skip_if_absent=True)
