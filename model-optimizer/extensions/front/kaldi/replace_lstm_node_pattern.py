"""
 Copyright (c) 2017-2018 Intel Corporation

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
import copy

import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.front.kaldi.extractor import common_kaldi_fields
from mo.front.kaldi.utils import KaldiNode
from mo.graph.graph import Node, unique_id as unique_node_id


def unique_id(prefix: str = 'id') -> str:
    """
    Generates a unique id
    The optional string prefix can be specified.
    """
    index = len(unique_id.names)
    name = prefix
    while name in unique_id.names:
        name = '{}_{}'.format(prefix, index)
        index += 1
    unique_id.names.append(name)
    return name


unique_id.names = []


def create_node(graph: nx.MultiDiGraph, name: str, attrs: dict, inputs: tuple = (), out_indexes: tuple = ([0]),
                weights=None, biases=None):
    """
    Create node with name 'name' and attributes from 'attrs'.
    Incoming edges for the node creates from nodes with id from 'inputs'
    Outgoing edges for the node creates to nodes with id from 'out_indexes'
    :param graph: graph to operate on.
    :param name: name how to save added node.
    :param attrs: optional attributes to be set. Attributes of the node
    :param inputs: tuple of ids inputs nodes
    :param out_indexes: tuple of ids outputs nodes
    :param weights: np.array of weights
    :param biases: np.array of biases
    :return:
    """
    unique_name = unique_node_id(graph, '{}_'.format(name))
    layer = KaldiNode(unique_name)
    layer.set_weight(weights)
    layer.set_bias(biases)
    layer.set_attrs(attrs)

    graph.add_node(unique_name, pb=layer, kind='op')
    new_graph_node = Node(graph, unique_name)
    graph.node[unique_name].update(common_kaldi_fields(new_graph_node))

    edge_attrs = {
        'out': 0,
        'in': 0,
        'name': layer.name,
        'fw_tensor_debug_info': [('', layer.name)],  # debug anchor for a framework tensor name and port
        'in_attrs': ['in', 'name'],
        'out_attrs': ['out', 'name'],
        'data_attrs': ['fw_tensor_debug_info']
    }

    edges = []
    for index, noe_id in enumerate(inputs):
        attrs = copy.deepcopy(edge_attrs)
        attrs['fw_tensor_debug_info'] = [(Node(graph, noe_id).soft_get('name'), None)]
        if index < len(out_indexes):
            attrs['out'] = out_indexes[index]
        attrs['in'] = index
        edges.append((noe_id, new_graph_node.id, attrs))

    graph.add_edges_from(edges)

    return new_graph_node


class ReplaceLSTMNodePattern(FrontReplacementOp):
    op = "LSTMProjectedStreams"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        input_node = node.in_nodes()[0]
        out_node = node.out_node()

        memory_pair_input = unique_id('id')
        memory_pair_output = unique_id('id')
        # Input -> FullyConnected
        fc_layer_after_input = create_node(graph, 'input_fullyconnected',
                                           dict(type='FullyConnected',
                                                num_output=node.pb.gifo_x_weights_shape[0],
                                                bias_term=True),
                                           tuple([input_node.id]),
                                           weights=node.pb.gifo_x_weights,
                                           biases=node.pb.gifo_biases)

        prev_lstm_output_node = create_node(graph, 'prev_memory_output',
                                            dict(type='Memory', id=memory_pair_input, index=1, size=2))

        # *Memory(output) -> FullyConnected
        fc_layer_from_prev_state = create_node(graph, 'prev_memory_output_fullyconnected',
                                               dict(type='FullyConnected', num_output=node.pb.gifo_r_weights_shape[0],
                                                    bias_term=False),
                                               tuple([prev_lstm_output_node.id]),
                                               weights=node.pb.gifo_r_weights)

        # Memory -> FullyConnected  \
        #                           *Eltwise(sum)
        # Input -> FullyConnected   /
        join_input_prev_state_sum_node = create_node(graph, 'join_input_eltwise',
                                                     dict(type='Eltwise', operation='sum'),
                                                     tuple([fc_layer_from_prev_state.id, fc_layer_after_input.id]))

        # *Eltwise(sum) -> Split
        # it is split into 4 nodes: Act, Eltw*3
        # the following order is mandatory
        #       ___Tanh
        #      /
        # Split ---(2)Eltwise(sum)
        #     |\
        #     | \__(3)Eltwise(sum)
        #     |____(4)Eltwise(sum)
        split_joined_input = create_node(graph, 'join_input_split',
                                         dict(type='Split', axis=None, num_split=4),

                                         tuple([join_input_prev_state_sum_node.id]))

        prev_lstm_state_node = create_node(graph, 'prev_memory_state',
                                           dict(type='Memory', id=memory_pair_output, index=1, size=2))

        # *Memory(state) -> *ScaleShift(input)
        state_input_scaleshift_node = create_node(graph, 'input_scaleshift',
                                                  dict(type='ScaleShift', bias_term=False),
                                                  tuple([prev_lstm_state_node.id]),
                                                  weights=node.pb.input_gate_weights)

        # *Memory(state) -> *ScaleShift(forget)
        state_forget_scaleshift_node = create_node(graph, 'forget_scaleshift',
                                                   dict(type='ScaleShift', bias_term=False),
                                                   tuple([prev_lstm_state_node.id]),
                                                   weights=node.pb.forget_gate_weights)

        # Split                                 \
        #                                       (2)Eltwise(sum)
        # Memory(state) -> *ScaleShift(input)  /
        join_prev_lstm_input_joined_input_sum_node = create_node(graph, 'join_prev_lstm_input_joined_input_eltwise',
                                                                 dict(type='Eltwise', operation='sum'),
                                                                 tuple([
                                                                     split_joined_input.id,
                                                                     state_input_scaleshift_node.id
                                                                 ]), out_indexes=(1, 0))

        # Split                                 \
        #                                       (3)Eltwise(sum)
        # Memory(state) -> *ScaleShift(forget)  /
        join_prev_lstm_input_joined_forget_sum_node = create_node(graph, 'join_prev_lstm_input_joined_forget_sum',
                                                                  dict(type='Eltwise', operation='sum'),
                                                                  tuple([
                                                                      split_joined_input.id,
                                                                      state_forget_scaleshift_node.id
                                                                  ]),
                                                                  out_indexes=(2, 0))

        # Split -> Tanh
        remember_tahn = create_node(graph, 'remember_tahn',
                                    dict(type='Activation', operation='tanh'),
                                    tuple([split_joined_input.id]), out_indexes=(0,))

        # Split -> (2)Eltwise(sum) -> *Sigmoid
        remember_sigmoid = create_node(graph, 'remember_sigmoid',
                                       dict(type='Activation', operation='sigmoid'),
                                       tuple([join_prev_lstm_input_joined_input_sum_node.id]))

        # Split -> (3)Eltwise(sum) -> **Sigmoid
        forget_sigmoid = create_node(graph, 'forget_sigmoid',
                                     dict(type='Activation', operation='sigmoid'),
                                     tuple([join_prev_lstm_input_joined_forget_sum_node.id]))

        # *Memory(state)                        \
        #                                       (6)Eltwise(mul)
        # Split -> (3)Eltwise(sum) -> **Sigmoid /
        join_forget_prev_state_mul_node = create_node(graph, 'join_forget_prev_state_mul',
                                                      dict(type='Eltwise', operation='mul'),
                                                      tuple([
                                                          forget_sigmoid.id,
                                                          prev_lstm_state_node.id
                                                      ]))

        # Split -> Tahn                         \
        #                                       (5)Eltwise(mul)
        # Split -> (2)Eltwise(sum) -> *Sigmoid   /
        join_remember_candidates_mul_node = create_node(graph, 'join_remember_candidates_mul',
                                                        dict(type='Eltwise', operation='mul'),
                                                        tuple([
                                                            remember_tahn.id,
                                                            remember_sigmoid.id,
                                                        ]))

        # (5)Eltwise(mul)  \
        #               (7)Eltwise(sum)
        # (6)Eltwise(mul)   /
        join_forget_remember_sum_node = create_node(graph, 'join_forget_remember_sum',
                                                    dict(type='Eltwise', operation='sum'),
                                                    tuple([
                                                        join_forget_prev_state_mul_node.id,
                                                        join_remember_candidates_mul_node.id,
                                                    ]))

        # (7)Eltwise(sum) -> Clamp
        join_forget_clamp_node = create_node(graph, 'join_forget_clamp',
                                             dict(type='Clamp', max=node.pb.clip_value, min=-node.pb.clip_value),
                                             tuple([join_forget_remember_sum_node.id]))

        # Clamp -> (2)Memory(state)
        next_lstm_state_node = create_node(graph, 'next_lstm_state',
                                           dict(type='Memory', id=memory_pair_output, index=0, size=2),
                                           tuple([join_forget_clamp_node.id]))

        # Clamp -> (2)Tahn
        state_filtered_tahn_node = create_node(graph, 'state_filtered_tahn',
                                               dict(type='Activation', operation='tanh'),
                                               tuple([join_forget_clamp_node.id]))

        # Clamp -> (2)ScaleShift
        clamp_scaleshift_node = create_node(graph, 'clamp_scaleshift',
                                            dict(type='ScaleShift', bias_term=False),
                                            tuple([join_forget_clamp_node.id]),
                                            weights=node.pb.output_gate_weights)

        # Split                 \
        #                       (4)Eltwise(sum)
        # Clamp -> (2)ScaleShift /
        join_next_lstm_input_joined_input_sum_node = create_node(graph, 'join_next_lstm_input_joined_input_sum',
                                                                 dict(type='Eltwise', operation='sum'),
                                                                 tuple([
                                                                     split_joined_input.id,
                                                                     clamp_scaleshift_node.id
                                                                 ]),
                                                                 out_indexes=(3, 0))

        # (4)Eltwise(sum) -> (3)Sigmoid
        output_sigmoid = create_node(graph, 'output_sigmoid',
                                     dict(type='Activation', operation='sigmoid'),
                                     tuple([join_next_lstm_input_joined_input_sum_node.id]))

        # (4)Eltwise(sum) -> (3)Sigmoid         \
        #                                       (5)Eltwise(mul)
        # Clamp -> (2)Tahn                      /
        joined_output_mul_node = create_node(graph, 'joined_output_mul',
                                             dict(type='Eltwise', operation='mul'),
                                             tuple([
                                                 state_filtered_tahn_node.id,
                                                 output_sigmoid.id
                                             ]))

        # (5)Eltwise(mul) -> (3)FullyConnected
        fc_output_node = create_node(graph, 'FullyConnected',
                                     dict(type='FullyConnected', num_output=node.pb.projection_weights_shape[0],
                                          bias_term=False),
                                     tuple([joined_output_mul_node.id]),
                                     weights=node.pb.projection_weights)

        #                   / (2)Memory(output)
        # (3)FullyConnected
        #                   \ Output (any next node) (edge created automatically after replacement)
        create_node(graph, 'next_lstm_output',
                    dict(type='Memory', id=memory_pair_input, index=0, size=2),
                    tuple([fc_output_node.id]))

        graph.remove_edges_from([input_node.id, node.id])
        graph.remove_edges_from([node.id, out_node.id])

        return [fc_output_node.id]
