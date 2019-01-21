"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np
from copy import deepcopy

from extensions.ops.lstm_sequence import LSTMSequence
from mo.utils.error import Error
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.op import Op
from mo.ops.split import Split
from mo.graph.graph import Node


class DecomposeBiLSTM(MiddleReplacementPattern):
    ''' Decomposes bidirectional LSTMSequence to forward and reverse LSTM ops.

        To extract forward and reverse parts from initial blobs, the helper
        functions used that should be already built-in into the operation attributes.

        Both initial state are split to two part, two parts of the results are concatenated.
        Axis of split/concat is completelly defined by ONNX/LSTM specification.
    '''

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(kind='op', op='LSTMSequence', format='onnx', direction='bidirectional')),
                ('input', dict(kind='data')),
                ('W', dict(kind='data')),
                ('R', dict(kind='data')),
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('W', 'lstm', {'bin': 'W'}),
                ('R', 'lstm', {'bin': 'R'}),
            ]
        )


    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        bilstm = match['lstm']
        new_init_hiddens = self.split_data(bilstm.in_node(5))
        new_init_cells = self.split_data(bilstm.in_node(6))
        assert bilstm.has_valid('blob_bidirectional_split'), \
            'Node {} doesnt\'t have blob_bidirectional_split attribute defined.'.format(bilstm.soft_get('name'))
        splitted_W = bilstm.blob_bidirectional_split(bilstm.in_node(1))
        splitted_R = bilstm.blob_bidirectional_split(bilstm.in_node(2))
        splitted_B = bilstm.blob_bidirectional_split(bilstm.in_node(3)) if 3 in bilstm.in_nodes() else (None, None)

        outputs = self.split_bilstm(
            bilstm,
            new_init_hiddens,
            new_init_cells,
            splitted_W,
            splitted_R,
            splitted_B,
        )

        self.concat(bilstm, outputs[0], outputs[1], bilstm.out_nodes())

    def split_data(self, data: Node):
        """ Split data node into two part along 0 axis """
        assert len(data.shape) == 3
        assert data.shape[0] == 2

        output_data = [Op._create_data_node(data.graph, name=data.name + '/SplittedBiLSTM/{}'.format(['forward', 'reverse'][i])) for i in [0, 1]]
        split_op = Split(data.graph, dict(name=data.name + '/DecomposedBiLSTM_0', axis=0, num_split=2))
        return split_op.create_node_with_data([data], data_nodes=output_data)


    def split_bilstm(self,
                     bilstm,
                     new_init_hiddens,
                     new_init_cells,
                     splitted_W,
                     splitted_R,
                     splitted_B):
        """ Split one bilstm node into 2 one-directional lstm nodes.

            All input data nodes should be already prepared; they are
            have 2 in the major dimension.
        """
        assert len(bilstm.out_nodes()) == 3
        all_outputs = []
        for i in [0, 1]:
            direction = ['forward', 'reverse'][i]
            op = LSTMSequence(bilstm.graph, {
                'hidden_size': bilstm.hidden_size,
                'direction': direction,
                'batch_dim': bilstm.batch_dim,
                'sequence_dim': bilstm.sequence_dim,
                'blobs_wrb': bilstm.blobs_wrb,
                'has_num_directions': bilstm.has_num_directions,
                'format': bilstm.format,
                'name': bilstm.name + '/Split/' + direction,
            })

            output_data = Op._create_data_node(
                bilstm.graph,
                name=bilstm.out_node(0).name + '/Split/' + str(i),
                attrs = {'shape': bilstm.out_node(0).shape.copy()}
            )

            assert output_data.shape[1] == 2
            output_data.shape[1] = 1

            output_hidden = Op._create_data_node(
                bilstm.graph,
                name=bilstm.out_node(1).name + '/Split/' + str(i),
                attrs = {'shape': bilstm.out_node(1).shape.copy()}
            )

            assert output_hidden.shape[0] == 2
            output_hidden.shape[0] = 1

            output_cell = Op._create_data_node(
                bilstm.graph,
                name=bilstm.out_node(2).name + '/Split/' + str(i),
                attrs = {'shape': bilstm.out_node(2).shape.copy()}
            )

            assert output_cell.shape[0] == 2
            output_cell.shape[0] = 1

            all_outputs.append(
                op.create_node_with_data(
                    inputs = [
                        bilstm.in_node(0),
                        splitted_W[i],
                        splitted_R[i],
                        splitted_B[i],
                        None,
                        new_init_hiddens[i],
                        new_init_cells[i],
                    ],
                    data_nodes = [
                        output_data,
                        output_hidden,
                        output_cell
                    ]
                )
            )
        return all_outputs


    def concat(self, bilstm, forward_outputs, reverse_outputs, final_outputs):
        """ Concatenates two set of outputs from BiLSTM """

        concat_ops = [
            Concat(bilstm.graph, {
                'name': bilstm.name + '/FinalConcat/Data',
                'axis': 1
            }),
            Concat(bilstm.graph, {
                'name': bilstm.name + '/FinalConcat/HiddenState',
                'axis': 0
            }),
            Concat(bilstm.graph, {
                'name': bilstm.name + '/FinalConcat/CellState',
                'axis': 0
            })
        ]

        bilstm.graph.remove_node(bilstm.id)

        for i in final_outputs:
            concat_ops[i].create_node_with_data(
                [forward_outputs[i], reverse_outputs[i]],
                data_nodes=[final_outputs[i]]
            )
