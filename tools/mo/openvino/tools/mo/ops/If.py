# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, is_fully_defined, dynamic_dimension_value, unmask_shape
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.op import Op


class If(Op):
    """
    If operation is an operation which has an input with condition which defines what sub-graph "then" or "else" to be
    executed.
    """
    op = 'If'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        base_attrs = {
            'type': self.op,
            'op': self.op,
            'then_graph': None,  # an Graph object with a "then" body sub-graph (condition is True)
            'else_graph': None,  # an Graph object with a "else" body sub-graph (condition is False)
            'sub_graphs': ['then_graph', 'else_graph'],  # built-in attribute with all sub-graphs
            'version': 'opset8',
            'infer': self.infer,
            'type_infer': self.type_infer,
        }
        base_attrs.update(attrs)
        super().__init__(graph, base_attrs, attrs)

    def port_map_attrs(self):
        return [
            'external_port_id',
            'internal_layer_id'
        ]

    @staticmethod
    def connect_body_input(if_node: Node, condition: bool, if_input_port_idx: int, body_parameter: Node):
        """
        Update the specified body parameter and connect it with If input

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_input_port_idx: the input port index to connect
        :param body_parameter: the body parameter node to connect
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert body_parameter.soft_get('op') == 'Parameter'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        assert body_parameter.id in sub_graph
        body_parameter['input_id'] = if_input_port_idx

    @staticmethod
    def connect_body_output(if_node: Node, condition: bool, if_output_port_idx: int, internal_result: Node):
        """
        Update the specified output port and connect it with If output

        :param if_node: the If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :param if_output_port_idx: the output port index to connect
        :param internal_result: the body Result node to connect
        :return: None
        """
        assert if_node.soft_get('op') == 'If'
        assert internal_result.soft_get('op') == 'Result'
        sub_graph = if_node.then_graph if condition else if_node.else_graph
        assert internal_result.id in sub_graph
        internal_result['output_id'] = if_output_port_idx

    @staticmethod
    def update_body_parameters_type(if_node: Node, condition: bool):
        """
        Update the data type for If body Parameter nodes based on data type of the outer graph nodes producing data
        for them.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph
        :return: None
        """
        assert if_node.soft_get('type') == 'If'

        subgraph = if_node.then_graph if condition else if_node.else_graph
        for node in subgraph.get_op_nodes():
            if node.has('input_id'):
                assert node.soft_get('type') == 'Parameter'
                input_port_id = node['input_id']
                input_type = if_node.in_port(input_port_id).get_data_type()
                node.data_type = input_type
                log.debug('Updated data type for the body node with name "{}" with value {}'
                          .format(node.name, node.data_type))

    @staticmethod
    def update_body_parameters_shape(if_node: Node, condition: bool):
        """
        Update shape for If body parameters.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph to add connect the body
        :return: None
        """
        subgraph = if_node.then_graph if condition else if_node.else_graph
        for node in subgraph.get_op_nodes():
            if node.has('input_id'):
                assert node.soft_get('type') == 'Parameter'
                input_port_id = node['input_id']
                input_shape = if_node.in_port(input_port_id).data.get_shape()
                if node.soft_get('shape', None) is None:
                    node['shape'] = None
                node.shape = input_shape.copy()
                log.debug('Updated shape for the body node with name "{}" with value {}'
                          .format(node.soft_get('name', node.soft_get('id')), node.shape))

    @staticmethod
    def results_mapping_and_finding_fake_outputs(output_nodes_in_subgraph, branch_name, outputs_mapping):
        """
        This method checked result nodes in subgraph and set map between output from If operation and internal subgraph
        result. Also This method return True if internal graph has fake results.

        :param output_nodes_in_subgraph: Result node with attribute 'output_id'
        :param branch_name: name of subgraph
        :param outputs_mapping: map between If operation output ID and subgraph results

        :return: True if all results of subgraph are empty tensors
        """
        graph_contain_fake_outputs = True

        for output_node in output_nodes_in_subgraph:
            assert output_node.soft_get('type') == 'Result'
            port_id = output_node['output_id']
            assert port_id in outputs_mapping.keys(), 'Incorrect mapping then_graph outputs with {0} outputs! ' \
                                                      'Can\'t find port with ID {1} in If operation.' \
                .format(output_node.name, port_id)
            outputs_mapping[port_id][branch_name] = output_node
            out_node_shape = output_node.in_port(0).data.get_shape()
            graph_contain_fake_outputs = graph_contain_fake_outputs and np.any(unmask_shape(out_node_shape) == 0)
        return graph_contain_fake_outputs

    @staticmethod
    def update_if_output_ports_shape(if_node: Node):
        """
        Update shape and values for If output ports.

        :param if_node: The If node to update output ports and shapes
        :return: None
        """
        node_name = if_node.soft_get('name', if_node.id)

        then_outputs = [node for node in if_node.then_graph.get_op_nodes() if node.has('output_id')]
        else_outputs = [node for node in if_node.else_graph.get_op_nodes() if node.has('output_id')]
        outputs_mapping = {}
        outputs_number = len(if_node.out_ports())

        if outputs_number == 0 and len(if_node.out_ports(control_flow=True)) != 0:
            # Some models have if with control flow outputs.
            # These shape inference for such ifs
            # TODO: need to rethink and redo support for control flow edges in if operation
            for node in if_node.out_nodes(control_flow=True).values():
                node.shape = int64_array([])
            return

        for port_id in if_node.out_ports().keys():
            outputs_mapping[port_id] = {}

        # variables then_contains_fake_outputs/else_contains_fake_outputs contains True value
        # if all outputs from then_body/else_body have shape [0]. It means then_body/else_body does not return data
        # and further shape_inference for this branch is not possible.
        # TODO: exclude support fake_outputs from this code when we will support shape_inference with empty tensors

        then_contains_fake_outputs = \
            If.results_mapping_and_finding_fake_outputs(then_outputs, 'then_graph', outputs_mapping)
        else_contains_fake_outputs = \
            If.results_mapping_and_finding_fake_outputs(else_outputs, 'else_graph', outputs_mapping)

        # use_then_shape is True when else_body or when both bodies do not return data. If use_then_shape is True If's
        # outputs will have the same shapes as then_body results
        use_then_shape = else_contains_fake_outputs or not then_contains_fake_outputs

        cond_value = if_node.in_port(0).data.get_value()

        for port_id in outputs_mapping:
            then_else_nodes = outputs_mapping[port_id]
            assert 'then_graph' in then_else_nodes.keys(), 'then_graph does not connect with If.out_port[{0}] ' \
                                                           'in {1} node!'.format(port_id, node_name)
            assert 'else_graph' in then_else_nodes.keys(), 'else_graph does not connect with If.out_port[{0}] ' \
                                                           'in {1} node!'.format(port_id, node_name)

            then_shape = then_else_nodes['then_graph'].in_port(0).data.get_shape()
            then_value = then_else_nodes['then_graph'].in_port(0).data.get_value()
            else_shape = then_else_nodes['else_graph'].in_port(0).data.get_shape()
            else_value = then_else_nodes['else_graph'].in_port(0).data.get_value()

            if is_fully_defined(cond_value):
                if cond_value.item() is True:
                    if then_value is not None:
                        if_node.out_port(port_id).data.set_value(then_value)
                    else:
                        if_node.out_port(port_id).data.set_shape(then_shape)
                else:
                    if else_value is not None:
                        if_node.out_port(port_id).data.set_value(else_value)
                    else:
                        if_node.out_port(port_id).data.set_shape(else_shape)
            else:
                if then_contains_fake_outputs ^ else_contains_fake_outputs:
                    # if exactly one of the outputs is fake then use another one
                    if_node.out_port(port_id).data.set_shape(then_shape if use_then_shape else else_shape)
                else:
                    # find "intersection" which is equal to the dimension value if corresponding dimensions are equal
                    # and dynamic otherwise
                    assert len(then_shape) == len(else_shape), 'Ranks of "then" and "else" output tensors are ' \
                                                               'different for node {} for port {}'.format(node_name,
                                                                                                          port_id)
                    output_shape = [d1 if is_fully_defined(d1) and is_fully_defined(d2) and d1 == d2 else
                                    dynamic_dimension_value for d1, d2 in zip(then_shape, else_shape)]
                    if_node.out_port(port_id).data.set_shape(output_shape)


    @staticmethod
    def update_if_output_ports_type(if_node: Node):
        """
        Update types for If output ports.

        :param if_node: The If node to update output ports and types
        :return: None
        """
        then_outputs = [node for node in if_node.then_graph.get_op_nodes() if node.has('output_id')]
        else_outputs = [node for node in if_node.else_graph.get_op_nodes() if node.has('output_id')]
        outputs_mapping = {}
        outputs_number = len(if_node.out_ports())
        assert outputs_number == len(then_outputs), 'Incorrect number outputs in then_graph of If with ' \
                                                    'name {0}! then_graph must has {1} outputs' \
            .format(if_node.name, outputs_number)
        assert outputs_number == len(else_outputs), 'Incorrect number outputs in else_graph of If with ' \
                                                    'name {0}! else_graph must has {1} outputs' \
            .format(if_node.name, outputs_number)
        for port_id in if_node.out_ports().keys():
            outputs_mapping[port_id] = {}
        port_ids = outputs_mapping.keys()
        for then_output_node in then_outputs:
            assert then_output_node.soft_get('type') == 'Result'
            port_id = then_output_node['output_id']
            assert port_id in port_ids, 'Incorrect mapping then_graph outputs with {0} outputs! ' \
                                        'Can\'t find port with ID {1} in If operation.' \
                .format(then_output_node.name, port_id)
            outputs_mapping[port_id]['then_graph'] = then_output_node

        for else_output_node in else_outputs:
            assert else_output_node.soft_get('type') == 'Result'
            port_id = else_output_node['output_id']
            assert port_id in port_ids, 'Incorrect mapping then_graph outputs with {0} outputs! ' \
                                        'Can\'t find port with ID {1} in If operation.' \
                .format(else_output_node.name, port_id)
            outputs_mapping[port_id]['else_graph'] = else_output_node

        for port_id in outputs_mapping:
            then_else_nodes = outputs_mapping[port_id]
            assert 'then_graph' in then_else_nodes.keys(), 'then_graph does not connect with If.out_port[{0}] ' \
                                                           'in {1} node!'.format(port_id, if_node.name)
            assert 'else_graph' in then_else_nodes.keys(), 'else_graph does not connect with If.out_port[{0}] ' \
                                                           'in {1} node!'.format(port_id, if_node.name)
            then_type = then_else_nodes['then_graph'].in_port(0).get_data_type()
            else_type = then_else_nodes['else_graph'].in_port(0).get_data_type()
            assert then_type == else_type, 'Cannot get type for if.out_port[{0}]! ' \
                                           'Types in then_graph and else_graph are not equal!'.format(port_id)
            if_node.out_port(port_id).set_data_type(then_type)

    @staticmethod
    def re_numerate_internal_id_and_get_if_id(if_node):
        """
        This method is called before IR generation. This method sets internal_layer_id.

        :param if_node: The If node where is necessary to set internal_layer_id in bodies.
        :return: if_node
        """
        then_graph_nodes = if_node.then_graph.nodes()
        for node in if_node.then_graph.get_op_nodes():
            then_graph_nodes[node.id]['internal_layer_id'] = node.id
        else_graph_nodes = if_node.else_graph.nodes()
        for node in if_node.else_graph.get_op_nodes():
            else_graph_nodes[node.id]['internal_layer_id'] = node.id
        return if_node.node

    def substitute_ie_attrs(self, new_attrs: dict):
        """
        Replace standard list of attribute in layer/data by attributes
        delivered by backend_attrs
        """

        port_map_attrs = self.port_map_attrs()
        new_attrs.update({
            'IE': [(
                'layer',
                [('id', lambda node: self.re_numerate_internal_id_and_get_if_id(node)), 'name', 'type', 'version'],
                [
                    '@ports',
                    ('then_port_map', [], [
                        ('@list', lambda node: self.generate_port_map(node, True, 'in'),
                         ('input', port_map_attrs, [])),
                        ('@list', lambda node: self.generate_port_map(node, True, 'out'),
                         ('output', port_map_attrs, [])),
                    ]),
                    ('else_port_map', [], [
                        ('@list', lambda node: self.generate_port_map(node, False, 'in'),
                         ('input', port_map_attrs, [])),
                        ('@list', lambda node: self.generate_port_map(node, False, 'out'),
                         ('output', port_map_attrs, [])),
                    ]),
                    ('then_body', [], [('@network', 'then_graph')]),
                    ('else_body', [], [('@network', 'else_graph')]),
                ])]
        })

    @staticmethod
    def generate_port_map(if_node: Node, condition: bool, dir: str):
        """
        Extract port_map attributes from if_node and its subgraphs attributes.

        :param if_node: The If node
        :param condition: the boolean defining a condition (then/else) graph
        :param dir: the str value defining type (for inputs or for putputs) of port_map
        :return: port_map -> list of dictionaries with to values(external_port_id or internal_layer_id)
        """
        port_map = []
        subgraph = if_node.then_graph if condition else if_node.else_graph
        name_of_connection = 'input_id' if dir == 'in' else 'output_id'

        for internal_node in subgraph.get_op_nodes():
            if internal_node.has(name_of_connection):
                port_map.append({'external_port_id': internal_node[name_of_connection],
                                 'internal_layer_id': internal_node['internal_layer_id']})

        return port_map

    @staticmethod
    def infer(if_node: Node):
        If.update_body_parameters_shape(if_node, True)
        If.update_body_parameters_shape(if_node, False)
        partial_infer(if_node.then_graph)
        partial_infer(if_node.else_graph)
        If.update_if_output_ports_shape(if_node)

    @staticmethod
    def type_infer(if_node: Node):
        from openvino.tools.mo.middle.passes.infer import type_infer
        If.update_body_parameters_type(if_node, True)
        If.update_body_parameters_type(if_node, False)
        type_infer(if_node.then_graph)
        type_infer(if_node.else_graph)
        If.update_if_output_ports_type(if_node)
