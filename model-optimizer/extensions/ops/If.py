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
import logging as log

from mo.graph.graph import Node, Graph
from mo.middle.passes.infer import partial_infer
from mo.ops.op import Op


class If(Op):
    """
    If operation is an operation which has an input with condition which defines what sub-graph "then" or "else" to be
    executed.
    """
    op = 'If'

    def __init__(self, graph: Graph, attrs: dict):
        base_attrs = {
            'type': self.op,
            'op': self.op,
            'then_input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'else_input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'then_output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'else_output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'back_edges': [],  # a list of dicts with such attrs as from_layer, from_port, etc.
            'then_graph': None,  # an Graph object with a "then" body sub-graph (condition is True)
            'else_graph': None,  # an Graph object with a "else" body sub-graph (condition is False)
            'sub_graphs': ['then_graph', 'else_graph'],  # built-in attribute with all sub-graphs
            'infer': self.infer,
            'type_infer': self.type_infer,
        }
        base_attrs.update(attrs)
        super().__init__(graph, base_attrs, attrs)

    def backend_attrs(self):
        return ['gg']

    def port_map_attrs(self):
        return [
            'external_port_id',
            'internal_layer_id',
            # 'internal_port_id',
        ]

    @staticmethod
    def connect_body_input(if_node: Node, condition: bool, if_input_port_idx: int, body_parameter: Node,
                           axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                           stride: [int, None] = None, part_size: [int, None] = None):
        """
        Update the input port map to connect the input port with the specified body parameter

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_input_port_idx: the input port index to connect
        :param body_parameter: the body parameter node to connect
        :param axis: dimension for input slicing
        :param start: start value of dimension from which to start slicing
        :param end: end value of dimension when to finish slicing
        :param stride: a step value for slicing
        :param part_size: a partial size for slicing, i.e. slicing [start; start + part_size)
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert body_parameter.soft_get('op') == 'Parameter'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        port_map = if_node.then_input_port_map if condition else if_node.else_input_port_map
        assert body_parameter.id in sub_graph

        port_map.append({'external_port_id': if_input_port_idx,
                         'internal_layer_id': body_parameter['internal_layer_id']})

    @staticmethod
    def connect_body_output(if_node: Node, condition: bool, if_output_port_idx: int, internal_result: Node,
                            axis: [int, None] = None, start: [int, None] = None, end: [int, None] = None,
                            stride: [int, None] = None, part_size: [int, None] = None):
        """
        Update the output port map to connect the body Result node with the specified output port

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_output_port_idx: the output port index to connect
        :param internal_result: the body Result node to connect
        :param axis: dimension for output concatenation
        :param start: start value of dimension from which to start concatenation
        :param end: end value of dimension when to finish concatenation
        :param stride: a step value for concatenation
        :param part_size: a partial size for concatenation, i.e. concatenation [start; start + part_size)
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert internal_result.soft_get('op') == 'Result'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        port_map = if_node.then_output_port_map if condition else if_node.else_output_port_map
        assert internal_result.id in sub_graph

        port_map.append({'external_port_id': if_output_port_idx,
                         'internal_layer_id': internal_result['internal_layer_id']})

    @staticmethod
    def update_body_parameters_type(if_node: Node, condition: bool):
        """
        Update the data type for If body Parameter nodes based on data type of the outer graph nodes producing data
        for them.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph to add conne
        :return: None
        """
        assert if_node.soft_get('type') == 'If'
        port_map = if_node.then_input_port_map if condition else if_node.else_input_port_map
        for record in port_map:
            body_node = If.get_body_node_by_internal_id(if_node, condition, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                if_port_idx = record['external_port_id']
                if if_port_idx != -1:
                    input_type = if_node.in_port(if_port_idx).get_data_type()
                else:  # this is a current iteration number input node which is not connected to the Loop node
                    assert record['purpose'] == 'current_iteration'
                    input_type = np.int64

                body_node.data_type = input_type
                log.debug('Updated data type for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.data_type))

    @staticmethod
    def updated_body_parameters_shape(if_node: Node, condition: bool):
        """
        Update shape for If body parameters.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :return: None
        """
        port_map = if_node.then_input_port_map if condition else if_node.else_input_port_map
        for record in port_map:
            body_node = If.get_body_node_by_internal_id(if_node, condition, record['internal_layer_id'])
            # the Parameter may be removed because it was not used in the body, for example, the current iteration
            # number input
            if body_node is not None:
                assert body_node.soft_get('type') == 'Parameter'

                loop_port_idx = record['external_port_id']
                input_shape = if_node.in_port(loop_port_idx).get_connection().get_source().data.get_shape()
                body_node.shape = input_shape.copy()
                log.debug('Updated shape for the body node with internal_id "{}" with value {}'
                          ''.format(record['internal_layer_id'], body_node.shape))

    @staticmethod
    def updated_if_output_ports_shape(if_node: Node):
        """
        Update shape and values for If output ports. If the number of iterations is dynamic then the corresponding
        dimension for the scan outputs (having "axis" attribute) are set to 1 because MO cannot generate IR with
        undefined dimensions.

        :param if_node: The If node to update output ports and shapes
        :return: None
        """
        out_ports = if_node.out_ports().keys()
        for out_port in out_ports:
            then_internal_port = [record['internal_layer_id'] for record in if_node.then_output_port_map
                                  if record['external_port_id'] == out_port][0]
            else_internal_port = [record['internal_layer_id'] for record in if_node.else_output_port_map
                                  if record['external_port_id'] == out_port][0]

            then_body_node = If.get_body_node_by_internal_id(if_node, True, then_internal_port)
            else_body_node = If.get_body_node_by_internal_id(if_node, False, else_internal_port)
            assert then_body_node is not None
            assert then_body_node.soft_get('type') == 'Result'
            assert else_body_node is not None
            assert else_body_node.soft_get('type') == 'Result'

            then_shape = then_body_node.in_port(0).data.get_shape()
            else_shape = else_body_node.in_port(0).data.get_shape()
            # This assert excluded dynamism
            comp = then_shape == else_shape
            assert comp.all(), ""
            if_node.out_port(out_port, control_flow=True).data.set_shape(then_shape)

    @staticmethod
    def updated_if_output_ports_type(if_node: Node):
        """
        Update shape and values for If output ports. If the number of iterations is dynamic then the corresponding
        dimension for the scan outputs (having "axis" attribute) are set to 1 because MO cannot generate IR with
        undefined dimensions.

        :param if_node: The If node to update output ports and shapes
        :return: None
        """
        out_ports = if_node.out_ports().keys()
        for out_port in out_ports:
            then_internal_port = [record['internal_layer_id'] for record in if_node.then_output_port_map
                                  if record['external_port_id'] == out_port][0]
            else_internal_port = [record['internal_layer_id'] for record in if_node.else_output_port_map
                                  if record['external_port_id'] == out_port][0]
            then_body_node = If.get_body_node_by_internal_id(if_node, True, then_internal_port)
            else_body_node = If.get_body_node_by_internal_id(if_node, False, else_internal_port)
            assert then_body_node is not None
            assert then_body_node.soft_get('type') == 'Result'
            assert else_body_node is not None
            assert else_body_node.soft_get('type') == 'Result'

            then_type = then_body_node.in_port(0).get_data_type()
            else_type = else_body_node.in_port(0).get_data_type()
            assert then_type==else_type, "Output types of else({}) and then({}) branch is not equal for node {}".\
                format(else_type, then_type, if_node.soft_get('name'))
            if_node.out_port(out_port).set_data_type(then_type)

    def substitute_ie_attrs(self, new_attrs: dict):
        """
        Replace standard list of attribute in layer/data by attributes
        delivered by backend_attrs
        """

        port_map_attrs = self.port_map_attrs()

        back_edges_attrs = [
            ('from-layer', 'from_layer'),
            ('to-layer', 'to_layer'),
        ]

        new_attrs.update({
            'IE': [(
                'layer',
                [('id', lambda node: node.node), 'name', 'type', 'version'],
                [
                    '@ports',
                    ('then_port_map', [], [
                        ('@list', lambda node: self.generate_port_map(node, node.then_input_port_map, 'in'),
                         ('input', port_map_attrs, [])),
                        ('@list', lambda node: self.generate_port_map(node, node.then_output_port_map, 'out'),
                         ('output', port_map_attrs, [])),
                    ]),
                    ('else_port_map', [], [
                        ('@list', lambda node: self.generate_port_map(node, node.else_input_port_map, 'in'),
                         ('input', port_map_attrs, [])),
                        ('@list', lambda node: self.generate_port_map(node, node.else_output_port_map, 'out'),
                         ('output', port_map_attrs, [])),
                    ]),
                    ('then_body', [], [('@network', 'then_graph')]),
                    ('else_body', [], [('@network', 'else_graph')]),
                ])]
        })

    @staticmethod
    def generate_port_map(node: Node, src_port_map, dir: str):
        """ Extract port_map attributes from node and node.body attributes.

            It iterates over src_port_map and substitute external_port_id, internal_port_id and
            internal_layer_id by real values queried from node ports and node.body attributes.
        """
        result_list = []
        for map_item in src_port_map:
            result = dict(map_item)
            assert result is not map_item
            # result['external_port_id'] = __class__.find_port_id(node, result['external_port_id'], 'external_port_id')
            # result['internal_layer_id'] = __class__.find_internal_layer_id(node.body, result['internal_layer_id'])
            result_list.append(result)
        return src_port_map

    @staticmethod
    def find_port_id(node: Node, virtual_id: str, attr: str):
        attrs = node.edge({attr: virtual_id})[2]
        assert bool('in' in attrs) != bool('out' in attrs), attrs
        return attrs['in' if 'in' in attrs else 'out']

    @staticmethod
    def generate_back_edges(node: Node):
        ''' Extract back_edges attributes from node and node.body attributes. '''
        result_list = []
        for back_edge in node.back_edges:
            result = dict(back_edge)
            assert result is not back_edge
            result['from_layer'] = __class__.find_internal_layer_id(node.body, result['from_layer'])
            result['to_layer'] = __class__.find_internal_layer_id(node.body, result['to_layer'])
            result_list.append(result)
        return result_list


    @staticmethod
    def get_body_node_by_internal_id(if_node: Node, condition: bool, internal_id: int):
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        suitable_nodes = sub_graph.get_op_nodes(internal_layer_id=internal_id)
        assert len(suitable_nodes) <= 1, \
            'Expected 0 or 1 node with `internal_layer_id`={}, {} found'.format(internal_id, len(suitable_nodes))
        return suitable_nodes[0] if len(suitable_nodes) == 1 else None

    @staticmethod
    def infer(if_node: Node):
        If.updated_body_parameters_shape(if_node, True)
        If.updated_body_parameters_shape(if_node, False)
        partial_infer(if_node.then_graph)
        partial_infer(if_node.else_graph)
        If.updated_if_output_ports_shape(if_node)

    @staticmethod
    def type_infer(if_node: Node):
        from mo.middle.passes.infer import type_infer
        If.update_body_parameters_type(if_node, True)
        If.update_body_parameters_type(if_node, False)
        type_infer(if_node.then_graph)
        type_infer(if_node.else_graph)
        If.updated_if_output_ports_type(if_node)
