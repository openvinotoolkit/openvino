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
from mo.graph.port import Port
from mo.middle.passes.infer import partial_infer
from mo.ops.const import Const


class Loop(TensorIterator):
    """
    Loop layer that iterates over tensors and execute embedded sub-graph. The main difference from the TensorIterator is
    that Loop operation performs explicit slicing of data using special input called "current_iteration". Also the Loop
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

    def port_map_attrs(self):
        return super().port_map_attrs() + ['purpose']

    @staticmethod
    def generate_port_map(node: Node, src_port_map, dir: str):
        result_list = []
        for record in src_port_map:
            # do not update ids for not-connected output which is used in the Loop operation only
            if record['external_port_id'] != -1:
                if dir == 'out':  # increase the output port id by the number of input ports
                    # update the port id for proper generation of a "ports" section
                    record['external_port_id'] += len(node.in_ports())
            record['internal_layer_id'] = TensorIterator.find_internal_layer_id(node.body, record['internal_layer_id'])
            result_list.append(record)
        return result_list

    @staticmethod
    def get_body_node_by_internal_id(loop_node: Node, internal_id: int):
        suitable_nodes = loop_node.body.get_op_nodes(internal_layer_id=internal_id)
        assert len(suitable_nodes) <= 1, \
            'Expected 0 or 1 node with `internal_layer_id`={}, {} found'.format(internal_id, len(suitable_nodes))
        return suitable_nodes[0] if len(suitable_nodes) == 1 else None

    @staticmethod
    def updated_body_parameters_shape(loop_node: Node):
        """
        Update shape for Loop body parameters.
        The input shape of the "current_iteration" number input is handled separately because it is not connected with
        the Loop operation.

        :param loop_node: The Loop node
        :return: None
        """
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
        """
        Update shape and values for Loop output ports. If the number of iterations is dynamic then the corresponding
        dimension for the scan outputs (having "axis" attribute) are set to 1 because MO cannot generate IR with
        undefined dimensions.

        :param loop_node: The Loop node
        :return: None
        """
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
                    num_iters = Loop.iterations_count(loop_node)
                    if num_iters is None:
                        log.error('Dynamic number of iterations for Loop node "{}". Consider number to be 1 to be able'
                                  ' to generate the IR.'.format(loop_name), extra={'is_warning': True})
                        num_iters = 1
                    output_shape[concat_axis] = num_iters
                # MO does not support evaluation of Loop scan outputs with const values
                if concat_axis is None and output_value is not None:
                    loop_node.out_port(loop_port_idx).data.set_value(output_value)
                else:
                    loop_node.out_port(loop_port_idx).data.set_shape(output_shape)

    @staticmethod
    def iterations_count(loop_node: Node):
        """
        Try to determine the number of loop iterations. If we detect that the number is dynamic then return None.

        :param loop_node: Loop operation node
        :return: number of iterations or None if the number depends on runtime values.
        """
        assert loop_node.soft_get('type') == 'Loop'

        if loop_node.is_in_port_connected(1):
            execution_condition = loop_node.in_port(1).data.get_value()
            if execution_condition is None:  # dynamic execution condition
                return None
            execution_condition = execution_condition.item()
            if not execution_condition:  # 0 iterations
                return 0
        num_iterations = loop_node.in_port(0).data.get_value()
        if num_iterations is not None:
            num_iterations = num_iterations.item(0)
        return num_iterations

    @staticmethod
    def update_body_parameters_type(loop_node: Node):
        """
        Update the data type for Loop body Parameter nodes based on data type of the outer graph nodes producing data
        for them.

        :param loop_node: The Loop node
        :return: None
        """
        assert loop_node.soft_get('type') == 'Loop'
        for record in loop_node.input_port_map:
            body_node = Loop.get_body_node_by_internal_id(loop_node, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                loop_port_idx = record['external_port_id']
                if loop_port_idx != -1:
                    input_type = loop_node.in_port(loop_port_idx).get_data_type()
                else:  # this is a current iteration number input node which is not connected to the Loop node
                    assert record['purpose'] == 'current_iteration'
                    input_type = np.int64

                body_node.data_type = input_type
                log.debug('Updated data type for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.data_type))

    @staticmethod
    def update_loop_output_ports_type(loop_node: Node):
        """
        Update the data type for Loop output ports.

        :param loop_node: The Loop node
        :return: None
        """
        assert loop_node.soft_get('type') == 'Loop'
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
    def external_port_id_to_body_node(loop_node: Node, external_port_id: int, port_map: dict):
        """
        Return the body Node connected to the Loop port with number "external_port_id".

        :param loop_node: the Loop node
        :param external_port_id: the Loop node port idx
        :param port_map: the port_map value to look for external_port_id
        :return: the corresponding body Node
        """
        assert loop_node.soft_get('type') == 'Loop'
        body_graph = loop_node.body
        result_nodes = []
        for record in port_map:
            if record['external_port_id'] == external_port_id:
                result_nodes.extend(body_graph.get_op_nodes(internal_layer_id=record['internal_layer_id']))
        assert len(result_nodes) == 1, 'There should be just one body node for external port "{}", but there "{}"' \
                                       ''.format(external_port_id, len(result_nodes))
        return result_nodes[0]

    @staticmethod
    def connect_body_input(loop_input_port: Port, internal_parameter: Node, external_node_out_port: Port = None,
                           axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                           stride: [int, None] = None, part_size: [int, None] = None):
        loop_node = loop_input_port.node
        assert loop_node.soft_get('op') == 'Loop'
        assert loop_input_port.type == 'in'
        assert internal_parameter.soft_get('op') == 'Parameter'
        assert internal_parameter.id in loop_node.body

        if external_node_out_port is not None:
            assert loop_input_port.disconnected()
            assert external_node_out_port.node.id not in loop_node.body
            loop_input_port.connect(external_node_out_port)

        loop_node.input_port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start,
                                         'end': end, 'external_port_id': loop_input_port.idx,
                                         'internal_layer_id': internal_parameter['internal_layer_id']})

    @staticmethod
    def connect_body_output(loop_output_port: Port, internal_result: Node, external_node_input_ports: list = None,
                            axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                            stride: [int, None] = None, part_size: [int, None] = None):
        loop_node = loop_output_port.node
        assert loop_node.soft_get('op') == 'Loop'
        assert loop_output_port.type == 'out'
        assert internal_result.soft_get('op') == 'Result'
        assert internal_result.id in loop_node.body

        if external_node_input_ports is not None:
            assert loop_output_port.disconnected()
            assert all([port.node.id not in loop_node.body for port in external_node_input_ports])
            for port in external_node_input_ports:
                port.disconnect()
                loop_output_port.connect(port)
        loop_node.output_port_map.append({'axis': axis, 'stride': stride, 'part_size': part_size, 'start': start,
                                          'end': end, 'external_port_id': loop_output_port.idx,
                                          'internal_layer_id': internal_result['internal_layer_id']})

    @staticmethod
    def add_back_edge(loop_node: Node, internal_parameter: Node, internal_result: Node):
        assert internal_parameter.id in loop_node.body
        assert internal_parameter.soft_get('op') == 'Parameter'
        assert internal_result.id in loop_node.body
        assert internal_result.soft_get('op') == 'Result'

        loop_node.back_edges.append({'from_layer': internal_result['internal_layer_id'],
                                     'to_layer': internal_parameter['internal_layer_id'],
                                     'from_port': 0,
                                     'to_port': 0})

    @staticmethod
    def pull_constant_inputs_into_body(loop_node: Node):
        for port_idx, in_port in reversed(loop_node.in_ports().items()):
            # TODO add a check that the input does not correspond to execution_condition
            if not in_port.disconnected() and in_port.get_source().node.soft_get('type') == 'Const':
                original_const_node = in_port.get_source().node
                new_const_node = Const(loop_node.body, original_const_node.attrs()).create_node()

                body_parameter = Loop.external_port_id_to_body_node(loop_node, port_idx, loop_node.input_port_map)
                body_parameter.out_port(0).get_connection().set_source(new_const_node.out_port(0))
                loop_node.body.remove_nodes_from([body_parameter.id])
                loop_node.delete_input_port(port_idx)

    @staticmethod
    def update_port_map_value(port_map: dict, attr: str, original_value: int, new_value: int):
        matched = 0
        for record in port_map:
            if record[attr] == original_value:
                record[attr] = new_value
                matched += 1
        assert matched == 1, 'More than one record in the portmap for attr "{}" wil original value "{}"' \
                             ''.format(attr, original_value)

    @staticmethod
    def re_numerate_input_ports(loop_node: Node):
        """
        Update input ports ids to be consecutive from 0 to num_input_ports - 1 and update the port_map values of the
        Loop node.

        :param loop_node: the Loop node
        :return: None
        """
        def re_number_input_port(loop_node: Node, old_port_id: int, new_port_id: int):
            loop_node.add_input_port(new_port_id, skip_if_exist=True)
            loop_node.in_port(old_port_id).get_connection().set_destination(loop_node.in_port(new_port_id))
            Loop.update_port_map_value(loop_node.input_port_map, 'external_port_id', old_port_id, new_port_id)

        if len(loop_node.in_ports()) > 0:
            max_port_id = sorted(loop_node.in_ports().keys())[-1]
            new_port_id = 0
            for port_id in range(max_port_id + 1):
                if port_id in loop_node.in_ports():
                    if port_id != new_port_id:
                        re_number_input_port(loop_node, port_id, new_port_id)
                    new_port_id += 1

            for port_idx_to_remove in reversed(range(new_port_id, max_port_id + 1)):
                loop_node.delete_input_port(port_idx_to_remove)

    @staticmethod
    def re_numerate_output_ports(loop_node: Node):
        """
        Update output ports ids to be consecutive from 0 to num_output_ports - 1 and update the port_map values of the
        Loop node.

        :param loop_node: the Loop node
        :return: None
        """
        assert loop_node.soft_get('type') == 'Loop'

        def re_number_output_port(loop_node: Node, old_port_id: int, new_port_id: int):
            loop_node.add_output_port(new_port_id, skip_if_exist=True)
            loop_node.out_port(old_port_id).get_connection().set_source(loop_node.out_port(new_port_id))
            Loop.update_port_map_value(loop_node.output_port_map, 'external_port_id', old_port_id, new_port_id)

        if len(loop_node.out_ports()) > 0:
            max_port_id = sorted(loop_node.out_ports().keys())[-1]
            new_port_id = 0
            for port_id in range(max_port_id + 1):
                if port_id in loop_node.out_ports():
                    if port_id != new_port_id:
                        re_number_output_port(loop_node, port_id, new_port_id)
                    new_port_id += 1

            for port_idx_to_remove in reversed(range(new_port_id, max_port_id + 1)):
                loop_node.delete_output_port(port_idx_to_remove)

    @staticmethod
    def remove_unused_ops_from_port_map(loop_node: Node, port_map: dict, port_map_attr: str, dir: [None, str] = None):
        """
        Find unused operations in the Loop body referenced in the port_map and removes Loop ports connected to it.
        Loop input port with index 0 and 1 are mandatory so cannot be removed.
        Output ports of the Loop may not be connected so check for that case also and remove such an ops from the
        port_map. The only exception is the "execution_condition" output which is a mandatory.

        :param loop_node: the Loop node to update
        :param port_map: the port_map (input, output or back edges)
        :param port_map_attr: the port_map attribute containing the `internal_layer_id`
        :param dir: the direction of the port_map meaning 'in' or 'out' port of the Loop
        :return:
        """
        record_ids_to_remove = []
        for record_id, record in enumerate(port_map):
            if len(loop_node.body.get_op_nodes(internal_layer_id=record[port_map_attr])) == 0 or \
                    (dir == 'out' and record.get('purpose', "") != 'execution_condition' and
                     record['external_port_id'] not in loop_node.out_ports()):
                record_ids_to_remove.append(record_id)
        for record_id_to_remove in reversed(record_ids_to_remove):
            if dir in ['in', 'out']:
                port_to_remove = port_map[record_id_to_remove]['external_port_id']
                if port_to_remove != -1:
                    if dir == 'in':
                        if port_to_remove not in [0, 1]:  # input port 0 and 1 are mandatory for the Loop node
                            loop_node.delete_input_port(port_to_remove)
                    elif dir == 'out' and port_to_remove in loop_node.out_ports():
                        loop_node.delete_output_port(port_to_remove)
            del port_map[record_id_to_remove]

    @staticmethod
    def normalize_input_output_ports(loop_node: Node):
        """
        Remove inputs, outputs, back edges from the port map which are not used in the body and were removed by a
        graph clean up, for example, in case of not used current_iteration body Parameter. Then renumber input/output
        ports.

        :param loop_node: the Loop node to normalize
        :return: None
        """
        Loop.remove_unused_ops_from_port_map(loop_node, loop_node.input_port_map, 'internal_layer_id', 'in')
        Loop.remove_unused_ops_from_port_map(loop_node, loop_node.output_port_map, 'internal_layer_id', 'out')
        Loop.remove_unused_ops_from_port_map(loop_node, loop_node.back_edges, 'to_layer')

        # remove not connected input/output ports
        Loop.re_numerate_input_ports(loop_node)
        Loop.re_numerate_output_ports(loop_node)

    @staticmethod
    def infer(loop_node: Node):
        Loop.updated_body_parameters_shape(loop_node)
        partial_infer(loop_node.body)
        Loop.updated_loop_output_ports_shape_and_value(loop_node)

    @staticmethod
    def type_infer(loop_node: Node):
        from mo.middle.passes.infer import type_infer
        Loop.update_body_parameters_type(loop_node)
        type_infer(loop_node.body)
        Loop.update_loop_output_ports_type(loop_node)
