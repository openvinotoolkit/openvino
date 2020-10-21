"""
 Copyright (C) 2017-2020 Intel Corporation

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
import logging as log

import numpy as np

from extensions.ops.tensor_iterator import TensorIterator
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.infer import partial_infer


class Loop(TensorIterator):
    """
    Loop layer that iterates over tensors and execute embedded sub-graph. The main difference from the TensorIterator is
    that Loop operation performs implicit slicing of data using special input called "current_iteration". Also the Loop
    has special input determining the execution condition and special output producing execution condition for the next
    iteration.
    """
    op = 'Loop'

    def __init__(self, graph: Graph, attrs: dict):
        base_attrs = {
            'type': self.op,
            'op': self.op,
            'version': 'opset5',
            'input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'back_edges': [],  # a list of dicts with such attrs as from_layer, from_port, etc.
            'body': None,  # an Graph object with a body sub-graph
            'sub_graphs': ['body'],  # built-in attribute with all sub-graphs
            'infer': self.infer,
            'type_infer': self.type_infer,
        }
        base_attrs.update(attrs)
        super().__init__(graph, base_attrs)

    @staticmethod
    def get_body_node_by_internal_id(loop_node: Node, internal_id: int):
        suitable_nodes = loop_node.body.get_op_nodes(internal_layer_id=internal_id)
        assert len(suitable_nodes) <= 1, \
            'Expected 0 or 1 node with `internal_layer_id`={}, {} found'.format(internal_id, len(suitable_nodes))
        return suitable_nodes[0] if len(suitable_nodes) == 1 else None

    @staticmethod
    def updated_body_parameters_shape(loop_node: Node):
        for record in loop_node.input_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                input_shape = int64_array([])  # this is a current iteration number input shape
                loop_port_idx = record['external_port_id']
                if loop_port_idx != -1:
                    input_shape = loop_node.in_port(loop_port_idx).get_connection().get_source().data.get_shape()
                body_node.shape = input_shape
                log.debug('Updated shape for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.shape))

    @staticmethod
    def updated_loop_output_ports_shape_and_value(loop_node: Node):
        loop_name = loop_node.soft_get('name', loop_node.id)
        for record in loop_node.output_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            assert body_node is not None
            assert body_node.soft_get('type') == 'Result'

            loop_port_idx = record['external_port_id']
            if loop_port_idx != -1:  # the id = -1 for execution condition output which is not connected anywhere
                output_value = body_node.in_port(0).data.get_value()
                output_shape = body_node.in_port(0).data.get_shape()
                concat_axis = record['axis']
                if concat_axis is not None:
                    assert output_shape[concat_axis] == 1, 'Dimension for concatenation is not equal to 1 for scan ' \
                                                           'output for Loop node "{}" for loop output port "{}"'.\
                        format(loop_name, loop_port_idx)
                    output_shape[concat_axis] = Loop.iterations_count(loop_node)
                    assert output_shape[concat_axis] is not None, 'Dynamic number of iterations for Loop node "{}"' \
                                                                  ''.format(loop_name)
                # MO does not support evaluation of Loop scan outputs with const values
                if concat_axis is None and output_value is not None:
                    loop_node.out_port(loop_port_idx).data.set_value(output_value)
                else:
                    loop_node.out_port(loop_port_idx).data.set_shape(output_shape)

    @staticmethod
    def iterations_count(loop_node: Node):
        assert loop_node.soft_get('type') == 'Loop'

        if loop_node.is_in_port_connected(1):
            execution_condition = loop_node.in_port(1).data.get_value()
            if execution_condition is None:  # dynamic execution condition
                return None
            if not execution_condition:  # 0 iterations
                return 0
        num_iterations = loop_node.in_port(0).data.get_value()
        if num_iterations is not None:
            num_iterations = num_iterations.item(0)
        return num_iterations

    @staticmethod
    def updated_body_parameters_type(loop_node: Node):
        for record in loop_node.input_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                loop_port_idx = record['external_port_id']
                if loop_port_idx != -1:
                    input_type = loop_node.in_port(loop_port_idx).get_data_type()
                else:  # this is a current iteration number input type
                    assert record['purpose'] == 'current_iteration'
                    input_type = np.bool

                body_node.data_type = input_type
                log.debug('Updated data type for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.data_type))

    @staticmethod
    def updated_loop_output_ports_type(loop_node: Node):
        for record in loop_node.output_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            assert body_node is not None
            assert body_node.soft_get('type') == 'Result'

            loop_port_idx = record['external_port_id']
            if loop_port_idx != -1:  # the id = -1 for execution condition output which is not connected anywhere
                output_type = body_node.in_port(0).get_data_type()
                loop_node.out_port(loop_port_idx).set_data_type(output_type)

    @staticmethod
    def mark_current_iteration_parameter_node(loop_node: Node, body_parameter_node: Node):
        assert body_parameter_node.id in loop_node.body
        assert body_parameter_node.soft_get('op') == 'Parameter'
        assert body_parameter_node.has_valid('internal_layer_id')
        assert len(loop_node.body.get_op_nodes(purpose='current_iteration')) == 0

        loop_node.input_port_map.append({'axis': None, 'stride': None, 'part_size': None, 'start': None, 'end': None,
                                         'external_port_id': -1, 'purpose': 'current_iteration',
                                         'internal_layer_id': body_parameter_node['internal_layer_id']})

    @staticmethod
    def mark_execution_condition_result_node(loop_node: Node, body_result_node: Node):
        assert body_result_node.id in loop_node.body
        assert body_result_node.soft_get('op') == 'Result'
        assert body_result_node.has_valid('internal_layer_id')
        assert len(loop_node.body.get_op_nodes(purpose='execution_condition')) == 0

        loop_node.output_port_map.append({'axis': None, 'stride': None, 'part_size': None, 'start': None, 'end': None,
                                          'external_port_id': -1, 'purpose': 'execution_condition',
                                          'internal_layer_id': body_result_node['internal_layer_id']})

    @staticmethod
    def re_numerate_output_ports(loop_node: Node):
        def update_port_map(port_map: dict, old_port_id: int, new_port_id: int):
            for record in port_map:
                if record['external_port_id'] == old_port_id:
                    record['external_port_id'] = new_port_id

        def re_number_output_port(loop_node: Node, old_port_id: int, new_port_id: int):
            loop_node.add_output_port(new_port_id, skip_if_exist=True)
            loop_node.out_port(old_port_id).get_connection().set_source(loop_node.out_port(new_port_id))
            update_port_map(loop_node.output_port_map, old_port_id, new_port_id)

        if len(loop_node.out_ports()) > 0:
            max_port_id = sorted(loop_node.out_ports().keys())[-1]
            new_port_id = 0
            for port_id in range(max_port_id + 1):
                if port_id in loop_node.out_ports():
                    if port_id != new_port_id:
                        re_number_output_port(loop_node, port_id, new_port_id)
                    new_port_id += 1

            for port_to_remove_id in reversed(range(new_port_id, max_port_id + 1)):
                loop_node.delete_output_port(port_to_remove_id)

    @staticmethod
    def infer(node: Node):
        Loop.updated_body_parameters_shape(node)
        partial_infer(node.body)
        Loop.updated_loop_output_ports_shape_and_value(node)

    @staticmethod
    def type_infer(node: Node):
        from mo.middle.passes.infer import type_infer
        Loop.updated_body_parameters_type(node)
        type_infer(node.body)
        Loop.updated_loop_output_ports_type(node)
