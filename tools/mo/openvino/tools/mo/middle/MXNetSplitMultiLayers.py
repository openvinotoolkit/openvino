# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_insert, int64_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import Op


class MXNetSplitLayersToRNNSequence(MiddleReplacementPattern):
    """
        Split MXNet multilayer cell to multiple one-layers cells LSTM/GRU/RNN.
        Also concatenate output hiddens and cells states of this layers.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('rnn_layer', dict(kind='op', type='RNNSequence', format='mxnet', multilayers=True)),
                ('input', dict(kind='data')),
                ('params', dict(kind='data')),
            ],
            edges=[
                ('input', 'rnn_layer', {'in': 0}),
                ('params', 'rnn_layer', {'in': 1}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        output_states = self.split_multilayer_cell(graph, match)

        rnn_layer = match['rnn_layer']
        self.concat_output_states(graph, match, output_states)
        rnn_layer.graph.remove_node(rnn_layer.id)

    @staticmethod
    def get_new_cell(multilayer_cell: Node, number: int):
        cell_class = Op.get_op_class_by_name(multilayer_cell.op)
        new_cell = lambda graph, attrs: cell_class(graph, attrs)
        attrs = multilayer_cell.attrs().copy()
        new_attrs = {
            'num_layers': 1,
            'multilayers': False,
            'name': multilayer_cell.name + '/LayerSplittedLSTM/{}'.format(number),
        }
        attrs.update(new_attrs)
        return new_cell(multilayer_cell.graph, attrs)

    def split_multilayer_cell(self, graph: Graph, match: dict):
        """
        Split one multilayer type=RNNSequence cell to num_layers consecutive cells.
        All parameters splits to parts for new num_layers cells.
        """
        input = match['input']
        rnn_layer = match['rnn_layer']
        params = match['params'].value.copy()

        have_hidden = False
        if 2 in rnn_layer.in_nodes():
            hidden_state_value = rnn_layer.in_node(2).value
            have_hidden = True

        have_cell = False
        if 3 in rnn_layer.in_nodes():
            cell_state_value = rnn_layer.in_node(3).value
            have_cell = True

        direction = 2 if rnn_layer.has_num_directions else 1
        num_layers = rnn_layer.num_layers
        input_size = input.shape[2]
        bsize = (2 * rnn_layer.hidden_size * direction * num_layers) * rnn_layer.multiplier

        size = rnn_layer.hidden_size * direction * rnn_layer.multiplier
        first_layer_params_size = (input_size + rnn_layer.hidden_size + 2) * size
        other_layer_params_size = (rnn_layer.hidden_size * direction + rnn_layer.hidden_size + 2) * size
        assert params.size == (first_layer_params_size + (num_layers - 1) * other_layer_params_size)

        input_node = input
        params_layer_size_count = 0
        output_states = [[], []]

        param_w = params[0:len(params)-bsize]
        param_b = params[len(params) - bsize:]
        layer_bsize = (2 * rnn_layer.hidden_size * direction) * rnn_layer.multiplier

        for l in range(num_layers):
            params_layer_size = first_layer_params_size if l == 0 else other_layer_params_size

            layer_params_w = param_w[params_layer_size_count: params_layer_size_count +
                                                              (params_layer_size - layer_bsize)].copy()
            layer_params_b = param_b[layer_bsize*l: layer_bsize*l+layer_bsize].copy()
            layer_params = np.concatenate((layer_params_w, layer_params_b), axis=0)
            params_layer_size_count = params_layer_size_count + params_layer_size - layer_bsize

            op = self.get_new_cell(rnn_layer, l)
            name = str(rnn_layer.soft_get('name', rnn_layer.id))
            params_value_node = Const(
                rnn_layer.graph,
                dict(name=name + '/LayerSplittedParamsLSTM/{}/'.format(l), value=layer_params)
            ).create_node_with_data()

            if have_hidden:
                layer_hidden_state = hidden_state_value[l * direction: l * direction + direction]  # pylint: disable=possibly-used-before-assignment
                hidden_state_value_node = Const(
                    rnn_layer.graph,
                    dict(name=name + '/LayerSplittedHiddenState/{}/'.format(l), value=layer_hidden_state)
                ).create_node_with_data()
            else:
                hidden_state_value_node = None

            if have_cell:
                layer_cell_state = cell_state_value[l * direction: l * direction + direction]  # pylint: disable=possibly-used-before-assignment
                cell_state_value_node = Const(
                    rnn_layer.graph,
                    dict(name=name + '/LayerSplittedCellState/{}/'.format(l), value=layer_cell_state)
                ).create_node_with_data()
            else:
                cell_state_value_node = None

            if l < num_layers-1:
                output_data = Op._create_data_node(
                    rnn_layer.graph,
                    name=rnn_layer.out_node(0).name + '/LayerSplit/' + str(l),
                    attrs={'shape': rnn_layer.out_node(0).shape.copy()}
                )
            else:
                output_data = rnn_layer.out_node(0)

            # Output nodes creating:
            state_size = int64_array([input.shape[rnn_layer.batch_dim], rnn_layer.hidden_size])
            if rnn_layer.has_num_directions:
                state_size = shape_insert(state_size, 0, direction)

            output_hidden = Op._create_data_node(
                rnn_layer.graph,
                name=rnn_layer.out_node(1).name + '/LayerSplit/' + str(l),
                attrs={'shape': mo_array(state_size)}
            )

            current_data_nodes = [output_data, output_hidden]

            if rnn_layer.op == 'LSTM':
                output_cell = Op._create_data_node(
                    rnn_layer.graph,
                    name=rnn_layer.out_node(2).name + '/LayerSplit/' + str(l),
                    attrs={'shape': mo_array(state_size)}
                )
                current_data_nodes.append(output_cell)

            data_nodes = op.create_node_with_data(
                inputs=[
                    input_node,
                    params_value_node,
                    hidden_state_value_node,
                    cell_state_value_node
                ],
                data_nodes=current_data_nodes,
            )

            input_node = data_nodes[0]
            output_states[0].append(data_nodes[1])

            if rnn_layer.op =='LSTM':
                output_states[1].append(data_nodes[2])

        return output_states

    @staticmethod
    def concat_output_states(graph: Graph, match: dict, new_states: list):
        """ Concatenates output states from multilayer layer. """
        rnn_layer = match['rnn_layer']
        original_states = [rnn_layer.out_node(i) if i in rnn_layer.out_nodes() else None for i in [1, 2]]

        concat_ops = [
            Concat(rnn_layer.graph, {
                'name': rnn_layer.name + '/FinalLayerSplitConcat/HiddenState',
                'axis': -1
            }),
            Concat(rnn_layer.graph, {
                'name': rnn_layer.name + '/FinalLayerSplitConcat/CellState',
                'axis': -1
            })
        ]

        for i in range(len(original_states)):  # [0] or [0, 1]
            if original_states[i] is None:
                continue
            concat_ops[i].attrs.update({'in_ports_count': len(new_states[i])})
            concat_ops[i].create_node_with_data(inputs=new_states[i], data_nodes=[original_states[i]])
