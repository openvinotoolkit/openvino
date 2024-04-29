# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.WhileNormalize import WhileNormalize
from openvino.tools.mo.front.tf.custom_subgraph_call import skip_nodes_by_condition
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.middle.pattern_match import find_pattern_matches, inverse_dict
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.loop import Loop
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


def find_subgraph_match_to_pattern(graph: Graph, body_pattern: dict):
    """
    Finds sub-graph matches corresponding pattern in graph
    :param graph: a graph where to search for matched sub-graph
    :param body_pattern: a pattern
    :return: a list of sub-graph matches
    """
    matches = []
    for match in find_pattern_matches(graph, **body_pattern):
        match = inverse_dict(match)
        for k in match:
            match[k] = Node(graph, match[k])
        matches.append(match)

    return matches


class MapFNInputSlicing(FrontReplacementSubgraph):
    """
    The transformation handles inputs slicing in While loop created by TensorFlow 2 Map Function primitive
    (see tf.map_fn). It avoids TensorListFromTensor and TensorFlowGetItem operations and replaces the original
    sub-graph by adding axis attribute in Loop node for slicing inputs.
    The transformation is also applicable to TensorFlow 2 Keras Simple RNN, GRU, and LSTM layers.
    """
    enabled = True

    def run_before(self):
        return [WhileNormalize]

    @staticmethod
    def get_body_pattern():
        return dict(
            nodes=[('tensor_list', dict(op='Parameter')),
                   ('current_iteration', dict(op='Parameter')),
                   ('slicing', dict(op='TensorListGetItem')),
                   ('const_increment', dict(op='Const')),
                   ('increment_iteration', dict(op='Add')),
                   ('increment_iteration_identity', dict(op='Identity')),
                   ('increment_iteration_result', dict(op='Result'))],
            edges=[('tensor_list', 'slicing', {'in': 0}),
                   ('current_iteration', 'slicing', {'in': 1}),
                   ('const_increment', 'increment_iteration', {'in': 1}),
                   ('current_iteration', 'increment_iteration', {'in': 0}),
                   ('increment_iteration', 'increment_iteration_identity', {'in': 0}),
                   ('increment_iteration_identity', 'increment_iteration_result', {'in': 0})]
        )

    @staticmethod
    def get_body_pattern_without_identity():
        return dict(
            nodes=[('tensor_list', dict(op='Parameter')),
                   ('current_iteration', dict(op='Parameter')),
                   ('slicing', dict(op='TensorListGetItem')),
                   ('const_increment', dict(op='Const')),
                   ('increment_iteration', dict(op='Add')),
                   ('increment_iteration_result', dict(op='Result'))],
            edges=[('tensor_list', 'slicing', {'in': 0}),
                   ('current_iteration', 'slicing', {'in': 1}),
                   ('const_increment', 'increment_iteration', {'in': 1}),
                   ('current_iteration', 'increment_iteration', {'in': 0}),
                   ('increment_iteration', 'increment_iteration_result', {'in': 0})]
        )

    @staticmethod
    def transform_map_fn_input_slicing(external_match: dict, internal_match: dict):
        """
        Transforms TensorFlow 2 input slicing into use of axis attribute for input port of Loop node
        :param external_match: a match used for handling a part of the main graph responsible for input slicing
        :param internal_match: a match used for handling a part of the body graph responsible for input slicing
        """
        loop_node = external_match['while']
        unstack_node = external_match['unstack']
        body_graph = loop_node['body']

        tensor_list_get_item_node = internal_match['slicing']
        unstack_placeholder = internal_match['tensor_list']
        tensor_list_get_item_node_name = tensor_list_get_item_node.soft_get('name', tensor_list_get_item_node.id)

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

        # 2. process locality of Loop node in the main graph to avoid unsupported operations:
        # TensorListFromTensor, TensorListReserve, and TensorListStack
        # remove TensorListFromTensor and pass a tensor to Loop as is
        unstack_node.out_port(0).get_connection().set_source(unstack_node.in_port(0).get_connection().get_source())

    def find_and_replace_pattern(self, graph: Graph):
        for loop_node in graph.get_op_nodes(op='Loop'):
            loop_name = loop_node.soft_get('name', loop_node.id)
            body_graph = loop_node['body']
            body_pattern = MapFNInputSlicing.get_body_pattern()
            body_pattern_without_identity = MapFNInputSlicing.get_body_pattern_without_identity()
            internal_matches = find_subgraph_match_to_pattern(body_graph, body_pattern)
            internal_matches += find_subgraph_match_to_pattern(body_graph, body_pattern_without_identity)

            for internal_match in internal_matches:
                # check if TensorListGetItem from the body graph is connected with TensorListFromTensor
                # from the main graph. If yes, the transformation detects input slicing by this port
                # and can use Loop axis attribute
                unstack_node = Loop.get_external_nodes_by_internal_id(loop_node,
                                                                      internal_match['tensor_list'].internal_layer_id)
                unstack_node = unstack_node[0] if (len(unstack_node) == 1
                                                   and unstack_node[0].op == 'TensorListFromTensor') else None
                if unstack_node is None:
                    log.info("A sub-graph around the loop node {} does not match "
                             "TensorFlow 2 MapFN pattern for input slicing".format(loop_name))
                    continue

                external_match = {'while': loop_node,
                                  'unstack': unstack_node}
                # check that back edges connect correct Parameter and Result nodes in the body
                # check connections between body input ports and external inputs ports of Loop node
                if Loop.back_edge_exists(loop_node.back_edges,
                                         internal_match['increment_iteration_result'].internal_layer_id,
                                         internal_match['current_iteration'].internal_layer_id):
                    MapFNInputSlicing.transform_map_fn_input_slicing(external_match, internal_match)


class MapFNOutputConcatenation(FrontReplacementSubgraph):
    """
    The transformation handles inputs slicing in While loop created by TensorFlow 2 Map Function primitive
    (see tf.map_fn). It avoids TensorListReserve, TensorListStack, and TensorListSetItem operations and replaces
    the original sub-graph by adding axis attribute in Loop node for concatenation of intermediate output results.
    The transformation is also applicable to TensorFlow 2 Keras Simple RNN, GRU, and LSTM layers.
    """
    enabled = True

    def run_before(self):
        return [WhileNormalize]

    @staticmethod
    def get_body_pattern():
        return dict(
            nodes=[('container', dict(op='Parameter')),
                   ('current_iteration', dict(op='Parameter')),
                   ('const_increment', dict(op='Const')),
                   ('increment_iteration', dict(op='Add')),
                   ('increment_iteration_identity', dict(op='Identity')),
                   ('increment_iteration_result', dict(op='Result')),
                   ('concatenation', dict(op='TensorListSetItem')),
                   ('concatenation_identity', dict(op='Identity')),
                   ('concatenation_result', dict(op='Result')),
                   ],
            edges=[('const_increment', 'increment_iteration', {'in': 1}),
                   ('current_iteration', 'increment_iteration', {'in': 0}),
                   ('container', 'concatenation', {'in': 0}),
                   ('current_iteration', 'concatenation', {'in': 1}),
                   ('concatenation', 'concatenation_identity', {'in': 0}),
                   ('concatenation_identity', 'concatenation_result', {'in': 0}),
                   ('increment_iteration', 'increment_iteration_identity', {'in': 0}),
                   ('increment_iteration_identity', 'increment_iteration_result', {'in': 0})]
        )

    @staticmethod
    def get_body_pattern_without_identity():
        return dict(
            nodes=[('container', dict(op='Parameter')),
                   ('current_iteration', dict(op='Parameter')),
                   ('const_increment', dict(op='Const')),
                   ('increment_iteration', dict(op='Add')),
                   ('increment_iteration_result', dict(op='Result')),
                   ('concatenation', dict(op='TensorListSetItem')),
                   ('concatenation_result', dict(op='Result'))
                   ],
            edges=[('const_increment', 'increment_iteration', {'in': 1}),
                   ('current_iteration', 'increment_iteration', {'in': 0}),
                   ('container', 'concatenation', {'in': 0}),
                   ('current_iteration', 'concatenation', {'in': 1}),
                   ('concatenation', 'concatenation_result', {'in': 0}),
                   ('increment_iteration', 'increment_iteration_result', {'in': 0})
                   ]
        )

    @staticmethod
    def transform_map_fn_output_concatenation(external_match: dict, internal_match: dict):
        """
        Transforms TensorFlow 2 output concatenation into use of axis attribute for output port of Loop node
        :param external_match: a match used for handling a part of the main graph responsible for output concatenation
        :param internal_match: a match used for handling a part of the body graph responsible for output concatenation
        """
        loop_node = external_match['while']
        stack_node = external_match['stack']
        list_reserve_node = external_match['reserve']
        body_graph = loop_node['body']

        tensor_list_set_item_node = internal_match['concatenation']
        tensor_list_set_item_node_name = tensor_list_set_item_node.soft_get('name', tensor_list_set_item_node.id)
        list_result_node = internal_match['concatenation_result']

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

        # remove TensorListStack to by-pass the node since the result from the Loop node is already concatenated
        stack_node.out_port(0).get_connection().set_source(stack_node.in_port(0).get_connection().get_source())

        # disconnect ListReserve node because it is no longer needed for Loop
        list_reserve_node.out_port(0).disconnect()

        # connect a number of iterations with trip count that can be received from the second input of ListReserve
        # create a constant network with True value for execution_condition so that OV can ignore execution condition
        # and perform trip_counts iterations. This approach with known trip count value allows to avoid dynamism.
        loop_node.in_port(1).disconnect()
        list_reserve_node.in_port(1).get_source().connect(loop_node.in_port(1))
        for record in loop_node.output_port_map:
            if 'purpose' in record and record['purpose'] == 'execution_condition':
                exec_cond_layer_id = record['internal_layer_id']
                exec_cond_node = Loop.get_body_node_by_internal_id(loop_node, exec_cond_layer_id)
                const_true = Const(body_graph, {'value': mo_array(True, dtype=bool)}).create_node()
                exec_cond_node.in_port(0).get_connection().set_source(const_true.out_port(0))

        # remove back edge
        for record in loop_node.back_edges:
            if 'from_layer' in record and record['from_layer'] == list_result_node_layer_id:
                loop_node.back_edges.remove(record)

    def find_and_replace_pattern(self, graph: Graph):
        for loop_node in graph.get_op_nodes(op='Loop'):
            loop_name = loop_node.soft_get('name', loop_node.id)
            body_graph = loop_node['body']
            body_pattern = MapFNOutputConcatenation.get_body_pattern()
            body_pattern_without_identity = MapFNOutputConcatenation.get_body_pattern_without_identity()
            internal_matches = find_subgraph_match_to_pattern(body_graph, body_pattern)
            internal_matches += find_subgraph_match_to_pattern(body_graph, body_pattern_without_identity)

            for internal_match in internal_matches:
                # check if TensorListReserve from the main graph is connected with Parameter node from the body graph
                # that is assigned for storing intermediate output results of While Loop. If yes, the transformation
                # detects intermediate outputs concatenation by this port and can use Loop axis attribute
                reserve_node = Loop.get_external_nodes_by_internal_id(loop_node,
                                                                      internal_match['container'].internal_layer_id)
                reserve_node = reserve_node[0] if (len(reserve_node) == 1 and
                                                   reserve_node[0].op == 'TensorListReserve') else None
                if reserve_node is None:
                    log.info("A sub-graph around the loop node {} does not match "
                             "TensorFlow 2 MapFN pattern for intermediate outputs concatenation".format(loop_name))
                    continue
                stack_node = Loop.get_external_nodes_by_internal_id(
                    loop_node, internal_match['concatenation_result'].internal_layer_id)
                stack_node = stack_node[0] if len(stack_node) == 1 else None

                if stack_node is None:
                    log.info("A sub-graph around the loop node {} does not match "
                             "TensorFlow 2 MapFN pattern for intermediate outputs concatenation".format(loop_name))
                    continue

                # skip StopGradient node if it exists between While loop output port and TensorListStack operation
                stack_node = skip_nodes_by_condition(stack_node, lambda x: x.has_and_set('identity'), True)
                stack_node = stack_node if stack_node.op == 'TensorListStack' else None
                if stack_node is None:
                    log.info("A sub-graph around the loop node {} does not match "
                             "TensorFlow 2 MapFN pattern for intermediate outputs concatenation".format(loop_name))
                    continue

                external_match = {'while': loop_node,
                                  'reserve': reserve_node,
                                  'stack': stack_node}
                # check that back edges connect Parameter node (or container with intermediate output results)
                # and concatenation result produced by TensorListSetItem node
                if Loop.back_edge_exists(loop_node.back_edges, internal_match['concatenation_result'].internal_layer_id,
                                         internal_match['container'].internal_layer_id) and \
                        Loop.back_edge_exists(loop_node.back_edges,
                                              internal_match['increment_iteration_result'].internal_layer_id,
                                              internal_match['current_iteration'].internal_layer_id):
                    MapFNOutputConcatenation.transform_map_fn_output_concatenation(external_match, internal_match)


class TensorListOutputConcatenation(FrontReplacementSubgraph):
    """
    The transformation handles inputs slicing in While loop. It avoids TensorListPushBack, and EmptyTensorList
    operations and replaces the original sub-graph by adding axis attribute in Loop node for concatenation of
    intermediate output results.
    """
    enabled = True

    def run_before(self):
        return [WhileNormalize]

    @staticmethod
    def get_body_pattern():
        return dict(
            nodes=[('container', dict(op='Parameter')),
                   ('concatenation', dict(op='TensorListPushBack')),
                   ('concatenation_result', dict(op='Result'))
                   ],
            edges=[
                   ('container', 'concatenation', {'in': 0}),
                   ('concatenation', 'concatenation_result', {'in': 0}),
                   ]
        )

    @staticmethod
    def transform_tensor_list_output_concatenation(external_match: dict, internal_match: dict):
        """
        Transforms TensorFlow 2 output concatenation into use of axis attribute for output port of Loop node
        :param external_match: a match used for handling a part of the main graph responsible for output concatenation
        :param internal_match: a match used for handling a part of the body graph responsible for output concatenation
        """
        loop_node = external_match['while']
        empty_tensor_list_node = external_match['reserve']
        body_graph = loop_node['body']

        tensor_list_push_back_node = internal_match['concatenation']
        tensor_list_push_back_node_name = tensor_list_push_back_node.soft_get('name', tensor_list_push_back_node.id)
        list_result_node = internal_match['concatenation_result']

        # replace TensorListPushBack with Unsqueeze and use axis attribute for corresponding Result node
        # to concatenate results from different iterations
        unsqueeze_list_element = create_op_with_const_inputs(body_graph, Unsqueeze, {1: int64_array(0)},
                                                             {'name': tensor_list_push_back_node_name +
                                                                      '/TensorListPushBackUnsqueeze'})
        tensor_list_push_back_node.in_port(1).get_connection().set_destination(unsqueeze_list_element.in_port(0))
        tensor_list_push_back_node.out_port(0).get_connection().set_source(unsqueeze_list_element.out_port(0))
        rename_nodes([(tensor_list_push_back_node, tensor_list_push_back_node_name + '/AbandonedName'),
                      (unsqueeze_list_element, tensor_list_push_back_node_name)])
        list_result_node_layer_id = list_result_node.internal_layer_id
        Loop.update_port_map_value_ext(loop_node.output_port_map, 'internal_layer_id', list_result_node_layer_id,
                                       'axis', 0)

        # disconnect EmptyTensorList node because it is no longer needed for Loop
        empty_tensor_list_node.out_port(0).disconnect()

        loop_node.in_port(1).disconnect()
        empty_tensor_list_node.in_port(1).get_source().connect(loop_node.in_port(1))

        # remove back edge
        for record in loop_node.back_edges:
            if 'from_layer' in record and record['from_layer'] == list_result_node_layer_id:
                loop_node.back_edges.remove(record)

    def find_and_replace_pattern(self, graph: Graph):
        for loop_node in graph.get_op_nodes(op='Loop'):
            loop_name = loop_node.soft_get('name', loop_node.id)
            body_graph = loop_node['body']
            body_pattern = TensorListOutputConcatenation.get_body_pattern()
            internal_matches = find_subgraph_match_to_pattern(body_graph, body_pattern)

            for internal_match in internal_matches:
                # check if EmptyTensorList from the main graph is connected with Parameter node from the body graph
                # that is assigned for storing intermediate output results of While Loop. If yes, the transformation
                # detects intermediate outputs concatenation by this port and can use Loop axis attribute
                reserve_node = Loop.get_external_nodes_by_internal_id(loop_node,
                                                                      internal_match['container'].internal_layer_id)
                reserve_node = reserve_node[0] if (len(reserve_node) == 1 and
                                                   reserve_node[0].op == 'EmptyTensorList') else None
                if reserve_node is None:
                    log.info("A sub-graph around the loop node {} does not match "
                             "TensorFlow 2 EmptyTensorList->TensorListPushBack pattern for intermediate "
                             "outputs concatenation".format(loop_name))
                    continue

                external_match = {'while': loop_node,
                                  'reserve': reserve_node}
                # check that back edges connect Parameter node (or container with intermediate output results)
                # and concatenation result produced by TensorListPushBack node
                if Loop.back_edge_exists(loop_node.back_edges, internal_match['concatenation_result'].internal_layer_id,
                                         internal_match['container'].internal_layer_id):
                    TensorListOutputConcatenation.transform_tensor_list_output_concatenation(external_match,
                                                                                             internal_match)
