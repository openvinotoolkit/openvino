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
from copy import copy, deepcopy
import numpy as np
import logging as log
from extensions.ops.parameter import Parameter
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, dict_includes, Graph
from mo.graph.port import Port
from mo.middle.passes.infer import partial_infer
from mo.ops.const import Const
from mo.ops.op import Op
from mo.utils.error import Error
from extensions.ops.tensor_iterator import TensorIterator


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
            'sub_graphs': [],  # built-in attribute with all sub-graphs
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
        for record in loop_node.output_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            assert body_node is not None
            assert body_node.soft_get('type') == 'Result'

            loop_port_idx = record['external_port_id']
            if loop_port_idx != -1:  # the id = -1 for execution condition output which is not connected anywhere
                output_value = body_node.in_port(0).data.get_value()
                output_shape = body_node.in_port(0).data.get_shape()
                if output_value is not None:
                    loop_node.out_port(loop_port_idx).data.set_value(output_value)
                else:
                    loop_node.out_port(loop_port_idx).data.set_shape(output_shape)

    @staticmethod
    def updated_body_parameters_type(loop_node: Node):
        for record in loop_node.input_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                input_type = np.bool  # this is a current iteration number input shape
                loop_port_idx = record['external_port_id']
                if loop_port_idx != -1:
                    input_type = loop_node.in_port(loop_port_idx).get_data_type()
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
    def infer(node: Node):
        Loop.updated_body_parameters_shape(node)
        partial_infer(node.body)
        Loop.updated_loop_output_ports_shape_and_value(node)
        # TODO think about constant folding for scan outputs

    @staticmethod
    def type_infer(node: Node):
        from mo.middle.passes.infer import type_infer
        Loop.updated_body_parameters_type(node)
        type_infer(node.body)
        Loop.updated_loop_output_ports_type(node)
