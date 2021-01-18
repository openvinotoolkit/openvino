"""
 Copyright (C) 2017-2021 Intel Corporation

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

from extensions.front.tf.WhileNormalize import WhileNormalize
from extensions.ops.loop import Loop
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.middle.pattern_match import find_pattern_matches, inverse_dict
from mo.ops.const import Const
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


def find_intergraph_match(graph: Graph, ext_match: dict, reserve_port: int, unstack_port: int, stack_port: int):
    """
    The function checks that logic using TensorList operations corresponds to subsequent slicing of input and
    concatenation of intermediate results for output.
    If pattern is successfully found, it returns matching dictionary.
    """
    def back_edge_exists(back_edges_map, from_layer, to_layer, from_port=0, to_port=0):
        for back_edge in back_edges_map:
            if back_edge['from_layer'] == from_layer and back_edge['to_layer'] == to_layer \
                    and back_edge['from_port'] == from_port and back_edge['to_port'] == to_port:
                return True
        return False

    def inter_edge_exists(input_port_map, external_port_id, internal_layer_id):
        for input_port in input_port_map:
            if input_port['external_port_id'] == external_port_id and \
                    input_port['internal_layer_id'] == internal_layer_id:
                return True
        return False

    loop_node = ext_match['while']
    body_graph = loop_node['body']

    # check that a list of tensors are iterated subsequently in the body graph
    # that is functionally equivalent to slicing a tensor through a certain dimension
    body_pattern = dict(
        nodes=[('container', dict(op='Parameter')),
               ('tensor_list', dict(op='Parameter')),
               ('current_iteration', dict(op='Parameter')),
               ('slicing', dict(op='TensorListGetItem')),
               ('const_increment', dict(op='Const')),
               ('increment_iteration', dict(op='Add')),
               ('increment_iteration_identity', dict(op='Identity')),
               ('increment_iteration_result', dict(op='Result')),
               ('concatenation', dict(op='TensorListSetItem')),
               ('concatenation_identity', dict(op='Identity')),
               ('concatenation_result', dict(op='Result')),
               ],
        edges=[('tensor_list', 'slicing', {'in': 0}),
               ('current_iteration', 'slicing', {'in': 1}),
               ('const_increment', 'increment_iteration', {'in': 1}),
               ('current_iteration', 'increment_iteration', {'in': 0}),
               ('container', 'concatenation', {'in': 0}),
               ('current_iteration', 'concatenation', {'in': 1}),
               ('concatenation', 'concatenation_identity', {'in': 0}),
               ('concatenation_identity', 'concatenation_result', {'in': 0}),
               ('increment_iteration', 'increment_iteration_identity', {'in': 0}),
               ('increment_iteration_identity', 'increment_iteration_result', {'in': 0}),
               ]
    )
    int_matches = []
    for match in find_pattern_matches(body_graph, **body_pattern):
        match = inverse_dict(match)
        for k in match:
            match[k] = Node(body_graph, match[k])
        int_matches.append(match)

    if len(int_matches) == 1:
        int_match = int_matches[0]
        # check that back edges connect correct Parameter and Result nodes in the body
        # check connections between body input ports and external inputs ports of Loop node
        # check connections between body output ports and external output ports of Loop node
        if not back_edge_exists(loop_node.back_edges, int_match['concatenation_result'].internal_layer_id,
                                int_match['container'].internal_layer_id) or \
                not back_edge_exists(loop_node.back_edges, int_match['increment_iteration_result'].internal_layer_id,
                                     int_match['current_iteration'].internal_layer_id) or \
                not inter_edge_exists(loop_node.input_port_map, reserve_port,
                                      int_match['container'].internal_layer_id) or \
                not inter_edge_exists(loop_node.input_port_map, unstack_port,
                                      int_match['tensor_list'].internal_layer_id) or \
                not inter_edge_exists(loop_node.output_port_map, stack_port,
                                      int_match['concatenation_result'].internal_layer_id):
            return None

        return int_match

    return None


def process_keras_rnn(graph: Graph, ext_match: dict, int_match: dict, strided_slice_port: int):
    """
    The function transforms the sub-graph so that no list of tensors presents, axis for Loop input ports performs slicing
    (TensorListFromTensor->TensorListGetItem), axis for Loop output port for concatenation of outputs generated on each
    iteration (TensorListSetItem->TensorListStack).
    """
    loop_node = ext_match['while']
    unstack_node = ext_match['unstack']
    stack_node = ext_match['stack']
    list_reserve_node = ext_match['reserve']
    body_graph = loop_node['body']

    tensor_list_get_item_node = int_match['slicing']
    unstack_placeholder = int_match['tensor_list']
    tensor_list_get_item_node_name = tensor_list_get_item_node.soft_get('name', tensor_list_get_item_node.id)
    tensor_list_set_item_node = int_match['concatenation']
    tensor_list_set_item_node_name = tensor_list_set_item_node.soft_get('name', tensor_list_set_item_node.id)
    list_result_node = int_match['concatenation_result']

    # 1. process the body graph to avoid unsupported operations: TensorListGetItem and TensorListSetItem
    # replace TensorListGetItem with Squeeze node and iterate through slices using axis for input port
    squeeze_list_element = create_op_with_const_inputs(body_graph, Squeeze, {1: int64_array(0)},
                                                       {'name': 'TensorListGetItemSqueeze'})
    tensor_list_get_item_node.in_port(0).get_connection().set_destination(squeeze_list_element.in_port(0))
    tensor_list_get_item_node.out_port(0).get_connection().set_source(squeeze_list_element.out_port(0))
    rename_nodes([(tensor_list_get_item_node, tensor_list_get_item_node_name + '/AbandonedName'),
                  (squeeze_list_element, tensor_list_get_item_node_name)])
    unstack_placeholder_layer_id = unstack_placeholder.internal_layer_id
    Loop.update_port_map_value_ext(loop_node.input_port_map, 'internal_layer_id', unstack_placeholder_layer_id,
                                   'axis', 0)

    # replace TensorListSetItem with Unsqueeze and use axis attribute for corresponding Result node
    # to concatenate results from different iterations
    unsqueeze_list_element = create_op_with_const_inputs(body_graph, Unsqueeze, {1: int64_array(0)},
                                                         {'name': 'TensorListSetItemUnsqueeze'})
    tensor_list_set_item_node.in_port(2).get_connection().set_destination(unsqueeze_list_element.in_port(0))
    tensor_list_set_item_node.out_port(0).get_connection().set_source(unsqueeze_list_element.out_port(0))
    rename_nodes([(tensor_list_set_item_node, tensor_list_set_item_node_name + '/AbandonedName'),
                  (unsqueeze_list_element, tensor_list_set_item_node_name)])
    list_result_node_layer_id = list_result_node.internal_layer_id
    Loop.update_port_map_value_ext(loop_node.output_port_map, 'internal_layer_id', list_result_node_layer_id,
                                   'axis', 0)

    # 2. process locality of Loop node in the main graph to avoid unsupported operations:
    # TensorListFromTensor, TensorListReserve, and TensorListStack
    # remove TensorListFromTensor and pass a tensor to Loop as is
    unstack_node.out_port(0).get_connection().set_source(unstack_node.in_port(0).get_connection().get_source())

    # remove TensorListStack to by-pass the node since the result from the Loop node is already concatenated
    stack_node.out_port(0).get_connection().set_source(stack_node.in_port(0).get_connection().get_source())

    # remove TensorListReserve generating Tensor container that is no longer needed
    graph.remove_node(list_reserve_node.id)

    # connect a number of iterations with maximum_iterations
    # create a constant network with True value for execution_condition so that IE can ignore execution condition
    # and perform trip_counts iterations. This allows to avoid dynamism
    loop_node.in_port(1).disconnect()
    loop_node.in_port(strided_slice_port).get_connection().add_destination(loop_node.in_port(1))
    for record in loop_node.output_port_map:
        if 'purpose' in record and record['purpose'] == 'execution_condition':
            exec_cond_layer_id = record['internal_layer_id']
            exec_cond_node = Loop.get_body_node_by_internal_id(loop_node, exec_cond_layer_id)
            const_true = Const(body_graph, {'value': True}).create_node()
            exec_cond_node.in_port(0).get_connection().set_source(const_true.out_port(0))


class KerasLSTMReplacer(FrontReplacementSubgraph):
    """
    TensorFlow 2 Keras LSTM operation is expressed through a sub-graph with While operation with a certain body graph.
    This replacer allows to detect such a sub-graph and to handle it by avoiding unsupported operations:
    TensorListFromTensor, TensorListReserve, and TensorListStack in the main graph;
    TensorListGetItem and TensorListSetItem in the body graph.
    """
    enabled = True

    def run_before(self):
        return [WhileNormalize]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('unstack', dict(op='TensorListFromTensor')),
                   ('reserve', dict(op='TensorListReserve')),
                   ('while', dict(op='Loop')),
                   ('stack', dict(op='TensorListStack')),
                   ],
            edges=[('reserve', 'while', {'in': 3}),
                   ('unstack', 'while', {'in': 7}),
                   ('while', 'stack', {'out': 3})]
        )

    def replace_sub_graph(self, graph: Graph, ext_match: dict):
        int_match = find_intergraph_match(graph, ext_match, 3, 7, 3)
        if int_match is not None:
            process_keras_rnn(graph, ext_match, int_match, 6)


class KerasGRUReplacer(FrontReplacementSubgraph):
    """
    TensorFlow 2 Keras GRU operation is expressed through a sub-graph with While operation with a certain body graph.
    This replacer allows to detect such a sub-graph and to handle it by avoiding unsupported operations:
    TensorListFromTensor, TensorListReserve, and TensorListStack in the main graph;
    TensorListGetItem and TensorListSetItem in the body graph.
    It transforms the sub-graph so that no list of tensors used, axis for Loop input ports performs slicing
    (TensorListFromTensor->TensorListGetItem), axis for Loop output port for concatenation of outputs generated on each
    iteration (TensorListSetItem->TensorListStack).
    """
    enabled = True

    def run_before(self):
        return [WhileNormalize]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('unstack', dict(op='TensorListFromTensor')),
                   ('reserve', dict(op='TensorListReserve')),
                   ('while', dict(op='Loop')),
                   ('stack', dict(op='TensorListStack')),
                   ],
            edges=[('reserve', 'while', {'in': 3}),
                   ('unstack', 'while', {'in': 6}),
                   ('while', 'stack', {'out': 3})]
        )

    def replace_sub_graph(self, graph: Graph, ext_match: dict):
        int_match = find_intergraph_match(graph, ext_match, 3, 6, 3)
        if int_match is not None:
            process_keras_rnn(graph, ext_match, int_match, 5)
