"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from extensions.front.tf.WhileNormalize import WhileNormalize
from mo.ops.unsqueeze import Unsqueeze
from mo.ops.squeeze import Squeeze
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.front.common.partial_infer.utils import int64_array
from extensions.ops.loop import Loop
from mo.ops.const import Const


class KerasRNNReplacer(FrontReplacementSubgraph):
    """
    ...
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

    def replace_sub_graph(self, graph: Graph, match: dict):
        loop_node = match['while']
        body_graph = loop_node['body']

        tensor_list_get_items = body_graph.get_op_nodes(op='TensorListGetItem')
        tensor_list_set_items = body_graph.get_op_nodes(op='TensorListSetItem')

        # process body graph by avoiding unsupported operations: TensorListGetItem and TensorListSetItem
        # remove TensorListGetItem node and iterate through slices using axis for input port
        tensor_list_get_item = tensor_list_get_items[0]
        list_placeholder_node = tensor_list_get_item.in_port(0).get_connection().get_source().node

        squeeze_list_element = create_op_with_const_inputs(body_graph, Squeeze, {1: int64_array(0)},
                                                           {'name': 'TensorListGetItemSqueeze'})
        tensor_list_get_item.in_port(0).get_connection().set_destination(squeeze_list_element.in_port(0))
        tensor_list_get_item.out_port(0).get_connection().set_source(squeeze_list_element.out_port(0))
        int_layer_id = list_placeholder_node.internal_layer_id

        # update input_port_map with axis for input
        for record in loop_node.input_port_map:
            if record['internal_layer_id'] == int_layer_id:
                record['axis'] = 0

        # replace TensorListSetItem with Unsqueeze and add axis attribute for corresponding Result node
        tensor_list_set_item = tensor_list_set_items[0]
        list_result_node = tensor_list_set_item.out_port(0).get_connection().get_destination().node
        list_result_node = list_result_node.out_port(0).get_connection().get_destination().node  # bypass identity node
        unsqueeze_list_element = create_op_with_const_inputs(body_graph, Unsqueeze, {1: int64_array(0)},
                                                             {'name': 'TensorListSetItemUnsqueeze'})
        tensor_list_set_item.in_port(2).get_connection().set_destination(unsqueeze_list_element.in_port(0))
        tensor_list_set_item.out_port(0).get_connection().set_source(unsqueeze_list_element.out_port(0))
        int_layer_id = list_result_node.internal_layer_id

        # update output_port_map with axis for input
        for record in loop_node.output_port_map:
            if record['internal_layer_id'] == int_layer_id:
                record['axis'] = 0

        # process locality of Loop node in the main graph to avoid unsupported operations:
        # TensorListFromTensor, TensorListReserve, and TensorListStack
        # remove TensorListFromTensor by passing input tensor as is
        unstack_node = match['unstack']
        unstack_node.out_port(0).get_connection().set_source(unstack_node.in_port(0).get_connection().get_source())

        # remove TensorListStack by passing input tensor as is
        stack_node = match['stack']
        stack_node.out_port(0).get_connection().set_source(stack_node.in_port(0).get_connection().get_source())

        # remove TensorListReserve generating Tensor container that is no longer needed
        # and remove corresponding Parameter node in the body graph
        list_reserve_node = match['reserve']
        graph.remove_node(list_reserve_node.id)

        # connect a number of iterations with maximum_iterations
        # to avoid dynamism
        loop_node.in_port(1).disconnect()
        loop_node.in_port(6).get_connection().add_destination(loop_node.in_port(1))
        #  boolean Result with condition execution
        for record in loop_node.output_port_map:
            if 'purpose' in record and record['purpose'] == 'execution_condition':
                result_int_id = record['internal_layer_id']
                result_node = Loop.get_body_node_by_internal_id(loop_node, result_int_id)
                const_true = Const(body_graph, {'value': np.array(1, dtype=bool)}).create_node()
                result_node.in_port(0).get_connection().set_source(const_true.out_port(0))

        #loop_node.in_port(6).get_source().get_connection().add_destination(loop_node.in_port(1))
        pass
