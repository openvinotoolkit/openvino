# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.LSTM import LSTM
from openvino.tools.mo.utils.error import Error


class BlockLSTMtoLSTMSequenceSingleFirstOutput(MiddleReplacementPattern):
    """
    This transformation handles BlockLSTM with just one output, concatenation of all the intermediate
    output values of the hidden.
    TODO: implement for non-constant weights and bias.
    For this, it requires to unbound from LSTMRNNSequenceToTensorIterator transformation
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.middle.LSTMRNNSequenceToTensorIterator import LSTMToTensorIterator
        return [LSTMToTensorIterator]

    def run_after(self):
        from openvino.tools.mo.middle.RNNSequenceNormalizeToIE import RNNSequenceNormalize
        return [RNNSequenceNormalize, BlockLSTMtoLSTMSequence]

    def pattern(self):
        return dict(
            nodes=[
                ('BlockLSTM', dict(op='BlockLSTM')),
                ('weights', dict(op='Const')),
                ('weights_data', dict(kind='data')),
                ('bias', dict(op='Const')),
                ('bias_data', dict(kind='data')),
            ],
            edges=[
                ('weights', 'weights_data'),
                ('weights_data', 'BlockLSTM', {'in': 1}),
                ('bias', 'bias_data'),
                ('bias_data', 'BlockLSTM', {'in': 2}),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        block_lstm = match['BlockLSTM']
        connected_out_ports = [port_idx for port_idx, port in block_lstm.out_ports().items() if
                               not port.disconnected()]
        # support only BlockLSTM with one output of concatenated hidden states from all steps
        if len(connected_out_ports) != 1 or connected_out_ports[0] != 0:
            return
        shift_const = block_lstm.forget_bias

        # determine hidden_size based on weights shapes
        w_shape = block_lstm.in_port(1).data.get_shape()
        b_shape = block_lstm.in_port(2).data.get_shape()
        hidden_size = dynamic_dimension
        if len(b_shape) > 0 and b_shape[0] is not dynamic_dimension:
            hidden_size = b_shape[0] // 4
        elif len(w_shape) > 1 and w_shape[1] is not dynamic_dimension:
            hidden_size = w_shape[1] // 4

        assert hidden_size is not dynamic_dimension, "OpenVINO does not support BlockLSTM with dynamic hidden_size."

        # normalize weights to the format required by OpenVINO LSTMSequence
        weights = block_lstm.in_port(1).data.get_value()
        biases = block_lstm.in_port(2).data.get_value()
        assert weights is not None and biases is not None, \
            "Internal Model Optimizer error: weights and bias values should be defined."
        # 1. reshape weights and bias to highlight channel dimension
        weights = weights.reshape([weights.shape[0], 4, hidden_size])
        biases = biases.reshape([4, hidden_size])
        # 2. reorder gates icfo --> fico for both weights and biases
        gate_reorder = [2, 0, 1, 3]
        weights = np.take(weights, gate_reorder, axis=1)
        biases = np.take(biases, gate_reorder, axis=0)
        # 3. shift_const.value should be added to the first 1/4th part of the biases (f-gate: 0)
        # Note: in case of moving this code up before gate reordering, the addition
        # should be applied at different place
        biases[0] += shift_const
        # 4. return to the original shapes
        weights = weights.reshape([weights.shape[0], -1])
        biases = biases.flatten()
        # 5. TF stores weights in IO, but OV requires it in OI: transpose
        weights = weights.transpose()
        # 6. set up normalized values for Constant weights and bias
        block_lstm.in_port(1).data.set_value(weights)
        block_lstm.in_port(2).data.set_value(biases)

        # re-number h_init_state, c_init_state input ports to match RNNSequence ports order
        # at this point there is no clear match to RNNSequence operations
        # next re-ordering is expected in LSTMRNNSequenceToTensorIterator transformation
        # to match LSTMCell inputs order
        init_hidden_state_source = block_lstm.in_port(3).get_source()
        init_cell_state_source = block_lstm.in_port(4).get_source()
        block_lstm.in_port(4).get_connection().set_source(init_hidden_state_source)
        block_lstm.add_input_port(5, skip_if_exist=True)
        block_lstm.in_port(5).get_connection().set_source(init_cell_state_source)
        block_lstm.delete_input_port(3, skip_if_absent=True)

        new_attrs = {'sequence_dim': 0,
                     'batch_dim': 1,
                     'direction': 'forward',
                     'hidden_size': hidden_size,
                     'format': 'tf',
                     }

        # update attributes of existing operation
        # input edges have "bin" attribute for LSTMRNNSequenceToTensorIterator
        LSTM.update_node_stat(block_lstm, new_attrs)


class BlockLSTMtoLSTMSequence(MiddleReplacementPattern):
    """
    MO virtual operation RNNSequence that converts to OV TensorIterator with LSTMCell inside supports 3 outputs:
    0: concatenated hidden states over the whole time sequence,
    1: last hidden state,
    2: last cell state.

    Replacer do several tasks:
    1. Checks if current BlockLSTM can be translated to IR (OV does not support concatenated cell state output
    which can be produced by BlockLSTM)
    2. Searches for sub-graph, that takes last cell state out of unsupported concatenated cell state output.
    We cut this sub-graph off in case if there are no other consumers of concatenated cell state output and we connect
    BlockLSTM to consumers of this sub-graph by port producing last cell state output
    3. Renumber input ports of BlockLSTM to match RNNSequence specification.
    4. (Optional. Resolves by multiple checks) We cut the same sug-graph (as in 2) for concatenated cell states check
    for better performance
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.middle.LSTMRNNSequenceToTensorIterator import LSTMToTensorIterator
        return [LSTMToTensorIterator]

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        from openvino.tools.mo.middle.RNNSequenceNormalizeToIE import RNNSequenceNormalize
        return [MiddleStart, RNNSequenceNormalize]

    def pattern(self):
        return dict(
            nodes=[
                ('BlockLSTM', dict(op='BlockLSTM')),

                # 0 port: output h vector over the whole time sequence
                ('concatenated_hidden_states', (dict(kind='data'))),

                ('mul', dict(op='Mul')),
                ('mul_data', dict(kind='data')),
                ('after_mul_op_to_the_rest_of_model', dict(kind='op')),
                ('concat_0', dict(op='ConcatV2')),
                ('concat_0_data', dict(kind='data')),
                ('reshape_0', dict(op='Reshape')),
                ('reshape_0_data', dict(kind='data')),
                ('gather_0', dict(op='Gather')),
                ('gather_0_data', dict(kind='data')),

                # 1 port: cell state before the tanh over the whole time sequence
                ('concatenated_cell_states_data', (dict(kind='data'))),

                ('concat_1', dict(op='ConcatV2')),
                ('concat_1_data', dict(kind='data')),
                ('reshape_1', dict(op='Reshape')),
                ('reshape_1_data', dict(kind='data')),
                ('gather_1', dict(op='Gather')),
                ('gather_1_data', dict(kind='data')),
            ],
            edges=[
                ('BlockLSTM', 'concatenated_hidden_states', {'out': 0}),
                ('concatenated_hidden_states', 'mul'),
                ('mul', 'mul_data'),
                ('mul_data', 'after_mul_op_to_the_rest_of_model'),
                ('mul_data', 'concat_0'),
                ('concat_0', 'concat_0_data'),
                ('concat_0_data', 'reshape_0'),
                ('reshape_0', 'reshape_0_data'),
                ('reshape_0_data', 'gather_0'),
                ('gather_0', 'gather_0_data'),

                ('BlockLSTM', 'concatenated_cell_states_data', {'out': 1}),
                ('concatenated_cell_states_data', 'concat_1', {'in': 1}),
                ('concat_1', 'concat_1_data'),
                ('concat_1_data', 'reshape_1'),
                ('reshape_1', 'reshape_1_data'),
                ('reshape_1_data', 'gather_1'),
                ('gather_1', 'gather_1_data')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        time_len = match['concatenated_hidden_states'].shape[0]
        r"""
        Working with concatenated_cell_states_data part first, because OV TensorIterator primitive doesn't have
        concatenated cell states output and if we can not collapse it, then we does not support this type of BlockLSTM

        We simplify the sub-graph below by taking another output of BlockLSTM:
        concatenated cell states over the whole time sequence -> last cell state

        BlockLSTM
           || out 1 (concatenated cell states coming out of BlockLSTM)
           \/  in 1
        ConcatV2
           || (concatenation with initial state or another unused data)
           \/
        Reshape
           ||
           \/
         Gather (taking the last cell state from previous BlockLSTM, if Gather indexes == time_len)
        """
        # check that there are no other consumers of concatenated_cell_states_data data flow
        valid_output_names = ['concat_1', 'concat_1_data', 'reshape_1', 'reshape_1_data', 'gather_1', 'gather_1_data']
        valid_output_node_ids = [match[name].id for name in valid_output_names]
        node_names_to_check_outputs = ['concatenated_cell_states_data', 'concat_1_data', 'reshape_1_data']
        for name in node_names_to_check_outputs:
            for node in match[name].out_nodes():
                if node.id not in valid_output_node_ids:
                    raise Error("BlockLSTM node {} has output which contains concatenated cell states over the whole "
                                "time sequence. It is not replaceable by another output and is not supported "
                                "originally".format(match['BlockLSTM'].id))

        # check that we really take the last cell state data by Gather
        gather_indexes = match['gather_1'].in_node(1).value
        if len(gather_indexes) == 1:
            gather_index = gather_indexes[0]
        else:
            raise Error("BlockLSTM node {} has output which contains concatenated cell states over the whole "
                        "time sequence. It is not replaceable by another output and is not supported "
                        "originally".format(match['BlockLSTM'].id))
        if gather_index != time_len:
            raise Error("BlockLSTM node {} has output which contains concatenated cell states over the whole "
                        "time sequence. It is not replaceable by another output and is not supported "
                        "originally".format(match['BlockLSTM'].id))

        """
        We passed #1 and #2 stages from class description. It means that we can translate the rest of the pattern 
        to LSTMSequence even without following optimizations
        """

        node = match['BlockLSTM']
        weights_node = node.in_node(1)
        biases_node = node.in_node(2)
        shift_const = node.forget_bias

        # Assign temporary shape for them for easier manipulation
        # TF stores weights in IO order
        input_size = node.in_node(0).shape[-1]
        hidden_size = node.in_node(3).shape[-1]
        weights = weights_node.value
        biases = biases_node.value
        assert weights.shape[0] == input_size + hidden_size, \
            "weights.shape={} input_size={} hidden_size={}".format(weights.shape, input_size, hidden_size)
        assert weights.shape[1] == biases.shape[0] == 4 * hidden_size, \
            "weights.shape={} biases.shape={} hidden_size={}".format(weights.shape, biases.shape, hidden_size)

        weights = weights.reshape([
            weights.shape[0],
            4,  # gates
            hidden_size
        ])

        biases = biases.reshape([
            4,  # gates
            hidden_size
        ])

        # Reorder gates icfo --> fico for both weights and biases
        gate_reorder = [2, 0, 1, 3]
        weights = np.take(weights, gate_reorder, axis=1)
        biases = np.take(biases, gate_reorder, axis=0)

        # shift_const.value should be added to the first 1/4th part of the biases (f-gate: 0)
        # Note: in case of moving this code up before gate reordering, the addition
        # should be applied at different place
        biases[0] += shift_const

        # Return to the original shapes
        weights = weights.reshape([weights.shape[0], -1])
        biases = biases.flatten()

        # TF stores weights in IO, but OV requires it in OI: transpose
        weights = weights.transpose()

        weights_node.value = weights
        weights_node.shape = int64_array(weights.shape)
        biases_node.value = biases
        biases_node.shape = int64_array(biases.shape)

        attrs = dict(graph.get_edge_data(match['gather_1'].id, match['gather_1_data'].id)[0])
        attrs.update({'out': 2})
        graph.remove_edge(match['BlockLSTM'].id, match['concatenated_cell_states_data'].id)
        graph.remove_edge(match['gather_1'].id, match['gather_1_data'].id)

        match['BlockLSTM'].add_output_port(attrs['out'])
        graph.add_edge(match['BlockLSTM'].id, match['gather_1_data'].id, **attrs)

        """
        #3 Renumbering h_init_state, c_init_state input ports to match RNNSequence ports order.
        """
        h_init_port = 4
        c_init_port = 5
        # c_init_state
        if 4 in node.in_nodes():
            assert c_init_port not in node.in_nodes()
            cell_state_edge = graph.get_edge_data(node.in_node(4).id, node.id)
            cell_state_edge[0]['in'] = c_init_port

        # h_init_state
        if 3 in node.in_nodes():
            assert h_init_port not in node.in_nodes()
            hidden_state_edge = graph.get_edge_data(node.in_node(3).id, node.id)
            hidden_state_edge[0]['in'] = h_init_port

        new_attrs = {'sequence_dim': 0,
                     'batch_dim': 1,
                     'direction': 'forward',
                     'hidden_size': match['concatenated_hidden_states'].shape[-1],
                     'format': 'tf',
                     }

        LSTM.update_node_stat(match['BlockLSTM'], new_attrs)

        """
        Optional #4 optimization from class description following
        """
        data_to_mul = [n for n in match['mul'].in_nodes().values() if n.id != match['concatenated_hidden_states'].id]
        if len(data_to_mul) != 1:
            return  # unexpected type of mul
        data_to_mul = data_to_mul[0]
        if not data_to_mul.has_valid('value'):
            return  # unexpected type of mul
        data_to_mul_value = data_to_mul.value
        if not np.all(data_to_mul_value == 1):
            return  # unexpected type of mul

        # remove useless mul
        attrs = dict(graph.get_edge_data(match['BlockLSTM'].id, match['concatenated_hidden_states'].id)[0])
        graph.remove_edge(match['BlockLSTM'].id, match['concatenated_hidden_states'].id)
        graph.remove_edge(match['mul'].id, match['mul_data'].id)
        graph.add_edge(match['BlockLSTM'].id, match['mul_data'].id, **attrs)

        # find true usages of concatenated hidden states data (not last hidden state)
        valid_output_names = ['mul_data', 'concat_0', 'concat_0_data', 'reshape_0', 'reshape_0_data', 'gather_0',
                              'gather_0_data']
        valid_output_node_ids = [match[name].id for name in valid_output_names]
        node_names_to_check_outputs = ['mul_data', 'concat_0_data', 'reshape_0_data']

        list_of_concatenated_hidden_states_children_node_ids = []
        for name in node_names_to_check_outputs:
            for node in match[name].out_nodes():
                if node.id not in valid_output_node_ids:
                    list_of_concatenated_hidden_states_children_node_ids.append(node.id)

        if len(list_of_concatenated_hidden_states_children_node_ids) != 1:
            return  # not supported placement of pattern
        conacenated_child_node_id = list_of_concatenated_hidden_states_children_node_ids[0]
        if conacenated_child_node_id != match['after_mul_op_to_the_rest_of_model'].id:
            return  # not supported placement of pattern

        gather_indexes = match['gather_0'].in_node(1).value
        if len(gather_indexes) == 1:
            gather_index = gather_indexes[0]
        else:
            return  # we have to translate this type of BlockLSTM to LSTMSequence to TensorIterator as is
        if gather_index != time_len:
            return  # we have to translate this type of BlockLSTM to LSTMSequence to TensorIterator as is

        attrs = dict(graph.get_edge_data(match['gather_0'].id, match['gather_0_data'].id)[0])
        attrs.update({'out': 1})
        graph.remove_edge(match['mul_data'].id, match['concat_0'].id)
        graph.remove_edge(match['gather_0'].id, match['gather_0_data'].id)

        graph.add_edge(match['BlockLSTM'].id, match['gather_0_data'].id, **attrs)
