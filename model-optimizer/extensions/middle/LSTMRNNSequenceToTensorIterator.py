"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import numpy as np

from extensions.middle.RNNSequenceNormalizeToIE import RNNSequenceNormalize
from extensions.ops.lstm_cell import LSTMCell
from extensions.ops.tensor_iterator import TensorIterator
from mo.graph.graph import Graph, add_opoutput
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class LSTMToTensorIterator(MiddleReplacementPattern):
    """ Converts normalized RNNSequence with op=LSTM to TensorIterator.

        Normalized RNNSequence means that it should be processed by
        RNNSequenceNormalize transform that ensures its strict form.

        This transformation builds an alternative sub-graph for LSTMSequence
        with TensorIterator connected in the same way as an original LSTMSequence
        node and with internal body represented as LSTMCell op node with necessary
        squeezes and unsqueezes around.
    """

    enabled = True
    force_clean_up = True
    id = 'lstm_to_tensor_iterator'
    
    def run_after(self):
        return [RNNSequenceNormalize]

    def run_before(self):
        from extensions.middle.permute_tensor_iterator import TransposeTensorIteratorLSTM
        return [TransposeTensorIteratorLSTM]

    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(kind='op', op='LSTM', type='RNNSequence')),
                ('input', dict(kind='data')),
                ('weights', dict(kind='data')),
                ('biases', dict(kind='data')),
                # don't capture optional input initial states here
                ('output', dict(kind='data')),
                # don't capture optional output last states here
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('weights', 'lstm', {'bin': 'weights', 'in': 1}),
                ('biases', 'lstm', {'bin': 'biases', 'in': 2}),
                ('lstm', 'output', {'out': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        lstm = match['lstm']

        # Build TensorIterator body first
        body = Graph(name=lstm.name + '/sub_graph')
        body.graph = graph.graph

        # 1. Input squeeze Reshape
        inputs = [Op._create_data_node(body, lstm.name + '/inport/' + str(inp),
                                       {'shape': lstm.in_node(inp).shape.copy(),
                                        'value': lstm.in_node(inp).value.copy()
                                        if lstm.in_node(inp).value is not None and inp in [1, 2] else None})
                  for inp in [0, 4, 5, 1, 2]]  # X, WR, B, h_init, c_init

        inputs[0].shape[lstm.sequence_dim] = 1
        reshape_dim = inputs[0].shape.copy()
        reshape_dim[lstm.batch_dim] = -1
        reshape_dim = np.delete(reshape_dim, lstm.sequence_dim)
        input_squeeze = Reshape(body, dict(name=lstm.name + '/input_squeeze', internal_layer_id=0))
        squeeze_dim_data = Const(body, {'name': lstm.name + '/input_squeeze_dim',
                                        'value': reshape_dim}).create_node_with_data()
        inputs[0] = input_squeeze.create_node_with_data([inputs[0], squeeze_dim_data],
                                                        edge_attrs=[{'internal_port_id': 0}])

        # 2. Output unsqueeze Reshape
        outputs = [Op._create_data_node(body, lstm.name + '/outport/' + str(out),
                                        {'shape': lstm.out_node(out).shape.copy() if out in lstm.out_nodes()
                                        else lstm.in_node(4).shape.copy()}) for out in [0, 1]]
        for out in outputs:
            add_opoutput(body, out.id, 0, False)

        unsqueezed_output_shape = outputs[0].shape.copy()
        unsqueezed_output_shape[lstm.sequence_dim] = 1
        squeezed_output_shape = np.delete(unsqueezed_output_shape, lstm.sequence_dim)
        outputs[0].shape = squeezed_output_shape
        unsqueezed_output_shape[lstm.batch_dim] = -1
        output_unsqueeze = Reshape(body, dict(name=lstm.name + 'output_unsqueeze', internal_layer_id=2))
        unsqueeze_dim_data = Const(body, {'name': lstm.name + '/output_unsqueeze_dim',
                                        'value': unsqueezed_output_shape}).create_node_with_data()

        # 3. LSTMCell
        lstm_cell_op = LSTMCell(body, dict(hidden_size=lstm.hidden_size,
                                           activations=lstm.activations,
                                           activation_alpha=lstm.activation_alpha,
                                           activation_beta=lstm.activation_beta,
                                           clip=lstm.clip,
                                           input_forget=lstm.input_forget,
                                           name=lstm.name + '/LSTMCell',
                                           internal_layer_id=1))
        lstm_cell_node = lstm_cell_op.create_node_with_data(inputs, data_nodes=outputs,
                                                            edge_attrs=[{}, {'internal_port_id': 1},
                                                                        {'internal_port_id': 2}, {'bin': 'weights'},
                                                                        {'bin': 'biases'}])
        lstm_cell_node[0].in_node().out_edge(0)['internal_port_id'] = 4
        lstm_cell_node[0].in_node().out_edge(1)['internal_port_id'] = 5
        lstm_cell_node[0] = output_unsqueeze.create_node_with_data([lstm_cell_node[0], unsqueeze_dim_data])
        lstm_cell_node[0].in_node().out_edge(0)['internal_port_id'] = 3
        add_opoutput(body, lstm_cell_node[0].id, 0, False)

        # 4. TensorIterator layer creating
        assert lstm.direction in ['forward', 'reverse']
        if lstm.direction == 'forward':
            stride = 1
            start = None
            end = None
        else:
            assert lstm.direction == 'reverse'
            stride = -1
            start = -1
            end = 0

        output_port_map = [{
            'external_port_id': 3,
            'internal_layer_id': 2,
            'internal_port_id': 3,

            'axis': lstm.sequence_dim,
            'stride': stride,
            'start': start,
            'end': end,
            'part_size': 1,
        }]

        # Adding h_state, c_state to outputs
        if len(lstm.out_nodes()) == 3:
            output_port_map.extend([{
                'external_port_id': 4,
                'internal_layer_id': 1,
                'internal_port_id': 4,
            }, {
                'external_port_id': 5,
                'internal_layer_id': 1,
                'internal_port_id': 5,
            }])

        ti_op = TensorIterator(graph, {
            'name': lstm.name + '/TensorIterator',
            'body': body,
            'in_ports_count': 3,
            'out_ports_count': len(lstm.out_nodes()),

            'input_port_map': [
                {
                    'external_port_id': 0,
                    'internal_layer_id': 0,
                    'internal_port_id': 0,

                    'axis': lstm.sequence_dim,
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
                {
                    'external_port_id': 2,
                    'internal_layer_id': 1,
                    'internal_port_id': 2,
                },
            ],

            'output_port_map': output_port_map,

            'back_edges': [
                {
                    'from_layer': 1,
                    'from_port': 4,
                    'to_layer': 1,
                    'to_port': 1,
                },
                {
                    'from_layer': 1,
                    'from_port': 5,
                    'to_layer': 1,
                    'to_port': 2,
                },
            ]
        })

        assert sorted(lstm.out_nodes().keys()) == list(range(len(lstm.out_nodes()))), \
            "There are gaps in output ports of LSTMSequence operation. Node {}".format(lstm.id)

        outs = ti_op.create_node_with_data([lstm.in_node(i) for i in [0, 4, 5]],  # X, h_init, c_init
                                           data_nodes=[lstm.out_node(i) for i in range(len(lstm.out_nodes()))],
                                           edge_attrs=[{'external_port_id': 0}, {'external_port_id': 1},
                                                       {'external_port_id': 2}])

        if not isinstance(outs, list):
            outs = list([outs])

        graph.remove_node(lstm.id)
        outs[0].in_edge(0)['external_port_id'] = 3
        for i, out in enumerate(outs[1:]):
            external_port_id = 4 + i
            out.in_edge()['external_port_id'] = external_port_id
