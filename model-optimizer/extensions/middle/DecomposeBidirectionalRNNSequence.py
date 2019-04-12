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

from mo.graph.graph import Node, Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.op import Op
from mo.ops.split import Split


class DecomposeBidirectionalRNNSequence(MiddleReplacementPattern):
    """
        Decomposes bidirectional RNNSequence to forward and reverse RNNSequence ops.

        Both initial state are split to two part, two parts of the results are concatenated.

        Axis of split/concat is completely defined by ONNX recurrent layers specification.
    """
    enabled = True

    def run_after(self):
        from extensions.middle.MXNetRNNSequenceNormalize import MXNetRNNSequenceNormalize
        from extensions.middle.ONNXRNNSequenceNormalize import ONNXRNNSequenceNormalize
        return [ONNXRNNSequenceNormalize, MXNetRNNSequenceNormalize]

    def pattern(self):
        return dict(
            nodes=[
                ('lstm', dict(kind='op', type='RNNSequence', direction='bidirectional')),
                ('input', dict(kind='data')),
                ('W', dict(kind='data')),
                ('R', dict(kind='data')),
                ('B', dict(kind='data')),
            ],
            edges=[
                ('input', 'lstm', {'in': 0}),
                ('W', 'lstm', {'in': 1}),
                ('R', 'lstm', {'in': 2}),
                ('B', 'lstm', {'in': 3}),
            ]
        )

    @staticmethod
    def split_helper(node: Node, index: int, direction: str, axis: int=0):
        return Op._create_data_node(
            node.graph,
            name=node.name + '/SplittedBiLSTM/{}/'.format(direction),
            attrs={'value': np.take(node.value, [index], axis),
                   'shape': np.array(np.take(node.value, [index], axis).shape, dtype=np.int64)}
        )

    def split_data(self, data: Node):
        """ Helper. Split data node into two part along 0 axis """
        assert len(data.shape) == 3
        assert data.shape[0] == 2

        output_data = [Op._create_data_node(data.graph,
                       name=data.name + '/SplittedBiLSTM/{}'.format(['forward', 'reverse'][i])) for i in [0, 1]]
        split_op = Split(data.graph, dict(name=data.name + '/DecomposedBiLSTM_0', axis=0, num_split=2,
                                          out_ports_count=2))
        return split_op.create_node_with_data([data], data_nodes=output_data)

    def replace_pattern(self, graph: Graph, match: dict):
        bidirectional_cell = match['lstm']
        new_init_hiddens = self.split_data(bidirectional_cell.in_node(5))
        new_init_cells = self.split_data(bidirectional_cell.in_node(6)) if 6 in bidirectional_cell.in_nodes()\
            else (None, None)

        blob_bidirectional_split = lambda node: (
            self.split_helper(node, 0, 'forward'),
            self.split_helper(node, 1, 'reverse')
        )

        splitted_W = blob_bidirectional_split(bidirectional_cell.in_node(1))
        splitted_R = blob_bidirectional_split(bidirectional_cell.in_node(2))
        splitted_B = blob_bidirectional_split(bidirectional_cell.in_node(3))

        outputs = self.split_bidirectional(
            bidirectional_cell,
            new_init_hiddens,
            new_init_cells,
            splitted_W,
            splitted_R,
            splitted_B,
        )

        self.concat_outputs(bidirectional_cell, outputs[0], outputs[1], bidirectional_cell.out_nodes())

    @staticmethod
    def get_new_cell(bidirectional_cell: Node, direction: str):
        assert direction in ['forward', 'reverse']

        cell_class = Op.get_op_class_by_name(bidirectional_cell.op)
        new_cell = lambda graph, attrs: cell_class(graph, attrs)
        attrs = bidirectional_cell.attrs().copy()
        new_attrs = {
            'direction': direction,
            'name': bidirectional_cell.name + '/Split/' + direction,
        }
        attrs.update(new_attrs)
        return new_cell(bidirectional_cell.graph, attrs)

    def split_bidirectional(self,
                            bidirectional_cell: Node,
                            new_init_hiddens: list,
                            new_init_cells: list,
                            splitted_W: tuple,
                            splitted_R: tuple,
                            splitted_B: tuple):
        """
            Split one bidirectional RNNSequence node into 2 one-directional RNNSequence nodes.

            All input data nodes should be already prepared; they are
            have 2 in the num_dir dimension.
        """
        all_outputs = []
        for i in [0, 1]:
            direction = ['forward', 'reverse'][i]
            op = self.get_new_cell(bidirectional_cell, direction)

            output_data = Op._create_data_node(
                bidirectional_cell.graph,
                name=bidirectional_cell.out_node(0).name + '/Split/' + str(i),
                attrs={'shape': bidirectional_cell.out_node(0).shape.copy()}
            )

            assert output_data.shape[1] == 2
            output_data.shape[1] = 1

            output_hidden = Op._create_data_node(
                bidirectional_cell.graph,
                name=bidirectional_cell.out_node(1).name + '/Split/' + str(i),
                attrs={'shape': bidirectional_cell.out_node(1).shape.copy()}
            )

            assert output_hidden.shape[0] == 2
            output_hidden.shape[0] = 1

            data_nodes = [
                output_data,
                output_hidden,
            ]

            if bidirectional_cell.op == 'LSTM':
                output_cell = Op._create_data_node(
                    bidirectional_cell.graph,
                    name=bidirectional_cell.out_node(2).name + '/Split/' + str(i),
                    attrs={'shape': bidirectional_cell.out_node(2).shape.copy()}
                )

                assert output_cell.shape[0] == 2
                output_cell.shape[0] = 1

                data_nodes.append(output_cell)

            all_outputs.append(
                op.create_node_with_data(
                    inputs=[
                        bidirectional_cell.in_node(0),
                        splitted_W[i],
                        splitted_R[i],
                        splitted_B[i],
                        None,
                        new_init_hiddens[i],
                        new_init_cells[i] if bidirectional_cell.op == 'LSTM' else None,
                    ],
                    data_nodes=data_nodes
                )
            )
        return all_outputs

    @staticmethod
    def concat_outputs(bi_rnn, forward_outputs, reverse_outputs, final_outputs):
        """ Concatenates two set of outputs from bidirectiondl RNNSequence nodes """
        concat_ops = [
            Concat(bi_rnn.graph, {
                'name': bi_rnn.name + '/FinalConcat/Data',
                'axis': 1,
                'in_ports_count': 2,
            }),
            Concat(bi_rnn.graph, {
                'name': bi_rnn.name + '/FinalConcat/HiddenState',
                'axis': 0,
                'in_ports_count': 2,
            }),
            Concat(bi_rnn.graph, {
                'name': bi_rnn.name + '/FinalConcat/CellState',
                'axis': 0,
                'in_ports_count': 2,
            })
        ]

        bi_rnn.graph.remove_node(bi_rnn.id)

        for i in final_outputs:
            concat_ops[i].create_node_with_data(
                [forward_outputs[i], reverse_outputs[i]],
                data_nodes=[final_outputs[i]]
            )
