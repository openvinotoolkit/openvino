# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.tensor_iterator import TensorIterator
from openvino.tools.mo.front.common.partial_infer.utils import shape_delete
from openvino.tools.mo.graph.graph import Graph, add_opoutput
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class GRUAndRNNToTensorIterator(MiddleReplacementPattern):
    """ Converts normalized RNNSequence with op=GRU/RNN to TensorIterator.

        Normalized RNNSequence means that it should be processed by
        RNNSequenceNormalize transform that ensures its strict form.

        This transformation builds an alternative sub-graph for GRUSequence
        with TensorIterator connected in the same way as an original GRUSequence
        node and with internal body represented as GRUCell op node with necessary
        squeezes and unsqueezes around.
    """

    enabled = True
    id = 'gru_and_rnn_to_tensor_iterator'

    def run_after(self):
        from openvino.tools.mo.middle.RNNSequenceNormalizeToIE import RNNSequenceNormalize
        return [RNNSequenceNormalize]

    def run_before(self):
        from openvino.tools.mo.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
        return [TransposeTensorIteratorLSTM]

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', dict(kind='op', type='RNNSequence')),
                ('input', dict(kind='data')),
                ('weights', dict(kind='data')),
                ('biases', dict(kind='data')),
                # don't capture optional input initial states here
                ('output', dict(kind='data')),
                # don't capture optional output last states here
            ],
            edges=[
                ('input', 'rnn_layer', {'in': 0}),
                ('weights', 'rnn_layer', {'bin': 'weights', 'in': 1}),
                ('biases', 'rnn_layer', {'bin': 'biases', 'in': 2}),
                ('rnn_layer', 'output', {'out': 0}),
            ]
        )

    @staticmethod
    def get_rnn_cell(name: str):
        op = Op.get_op_class_by_name(name + 'Cell')
        return op

    def replace_pattern(self, graph: Graph, match: dict):
        if match['rnn_layer']['op'] == 'LSTM':
            return

        rnn_layer = match['rnn_layer']

        # Build TensorIterator body first
        body = Graph(name=rnn_layer.name + '/sub_graph')
        body.graph = graph.graph

        # 1. Input squeeze Reshape
        inputs = [Op._create_data_node(body, rnn_layer.name + '/inport/' + str(inp),
                                       {'shape': rnn_layer.in_node(inp).shape.copy(),
                                        'value': rnn_layer.in_node(inp).value.copy()
                                        if rnn_layer.in_node(inp).value is not None and inp in [1, 2] else None})
                  for inp in [0, 4, 1, 2]]  # X, h_init, WR, B

        inputs[0].shape[rnn_layer.sequence_dim] = 1
        input_squeeze = Squeeze(body, dict(name=rnn_layer.name + '/input_squeeze', internal_layer_id=0))
        input_squeeze_dim = Const(body, dict(name=rnn_layer.name + '/input_squeeze_dim',
                                             value=rnn_layer.sequence_dim)).create_node_with_data()
        inputs[0] = input_squeeze.create_node_with_data([inputs[0], input_squeeze_dim],
                                                        edge_attrs=[{'internal_port_id': 0}])

        # 2. Output unsqueeze Reshape
        outputs = [Op._create_data_node(body, rnn_layer.name + '/outport/' + str(out),
                                        {'shape': rnn_layer.out_node(out).shape.copy() if out in rnn_layer.out_nodes() else None})
                                        for out in [0]]
        for out in outputs:
            add_opoutput(body, out.id, 0, False)

        outputs[0].shape = shape_delete(outputs[0].shape, rnn_layer.sequence_dim)
        output_unsqueeze_dim = Const(body, dict(name=rnn_layer.name + '/output_unsqueeze_dim',
                                                value=rnn_layer.sequence_dim)).create_node_with_data()
        output_unsqueeze = Unsqueeze(body, dict(name=rnn_layer.name + '/output_unsqueeze/', internal_layer_id=2))

        additional_attrs = dict(activations=rnn_layer.activations,
                                activation_alpha=rnn_layer.activation_alpha,
                                activation_beta=rnn_layer.activation_beta,
                                clip=rnn_layer.clip)
        if rnn_layer.op == 'GRU':
            additional_attrs['linear_before_reset'] = rnn_layer.linear_before_reset

        # 3. ***Cell
        rnn_cell_op = self.get_rnn_cell(rnn_layer['op'])(body, dict(hidden_size=rnn_layer.hidden_size,
                                                                    name=rnn_layer.name + '/{}Cell'.format(rnn_layer.op),
                                                                    **additional_attrs,
                                                                    internal_layer_id=1))

        gru_cell = rnn_cell_op.create_node_with_data(inputs, data_nodes=outputs,
                                                             edge_attrs=[{}, {'internal_port_id': 1},
                                                                        {'internal_port_id': 2}, {'bin': 'weights'},
                                                                        {'bin': 'biases'}])

        # internal ports for outputs of cell
        gru_cell.in_node().out_edge(0)['internal_port_id'] = 4  # h_state

        gru_cell = output_unsqueeze.create_node_with_data([gru_cell, output_unsqueeze_dim])
        gru_cell.in_node().out_edge(0)['internal_port_id'] = 3
        add_opoutput(body, gru_cell.id, 0, False)

        # 4. TensorIterator layer creating
        assert rnn_layer.direction in ['forward', 'reverse']
        if rnn_layer.direction == 'forward':
            stride = 1
            start = None
            end = None
        else:
            assert rnn_layer.direction == 'reverse'
            stride = -1
            start = -1
            end = 0

        # stacked h_state
        output_port_map = [{
            'external_port_id': 3,
            'internal_layer_id': 2,
            'internal_port_id': 3,

            'axis': rnn_layer.sequence_dim,
            'stride': stride,
            'start': start,
            'end': end,
            'part_size': 1,
        }]

        # Adding last h_state to outputs
        if len(rnn_layer.out_nodes()) == 2:
            output_port_map.extend([{
                'external_port_id': 4,
                'internal_layer_id': 1,
                'internal_port_id': 4,
            }])

        ti_op = TensorIterator(graph, {
            'name': rnn_layer.name + '/TensorIterator',
            'body': body,
            'in_ports_count': 4,
            'out_ports_count': len(rnn_layer.out_nodes()),

            'input_port_map': [
                {
                    'external_port_id': 0,
                    'internal_layer_id': 0,
                    'internal_port_id': 0,

                    'axis': rnn_layer.sequence_dim,
                    'stride': stride,
                    'start': start,
                    'end': end,
                    'part_size': 1,
                },
                {
                    'external_port_id': 1,
                    'internal_layer_id': 1,
                    'internal_port_id': 1,
                },
            ],

            'output_port_map': output_port_map,
            # only for h state
            'back_edges': [
                {
                    'from_layer': 1,
                    'from_port': 4,
                    'to_layer': 1,
                    'to_port': 1,
                },
            ]
        })

        assert sorted(rnn_layer.out_nodes().keys()) == list(range(len(rnn_layer.out_nodes()))), \
            "There are gaps in output ports of GRUSequence operation. Node {}".format(rnn_layer.id)

        outs = ti_op.create_node_with_data([rnn_layer.in_node(i) for i in [0, 4]],  # X, h_init
                                           data_nodes=[rnn_layer.out_node(i) for i in range(len(rnn_layer.out_nodes()))],
                                           edge_attrs=[{'external_port_id': 0}, {'external_port_id': 1}])

        if not isinstance(outs, list):
            outs = list([outs])

        graph.remove_node(rnn_layer.id)
        outs[0].in_edge(0)['external_port_id'] = 3
        for i, out in enumerate(outs[1:]):
            external_port_id = 4 + i
            out.in_edge()['external_port_id'] = external_port_id

        ti = outs[0].in_node()
        TensorIterator.cover_body_input_data_nodes_with_parameter_ops(ti)
        TensorIterator.cover_body_constant_data_nodes_with_const_ops(ti)
        TensorIterator.normalize_internal_ids(ti)
