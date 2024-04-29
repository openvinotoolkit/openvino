# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from copy import copy, deepcopy
from math import ceil

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, is_fully_defined, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node, dict_includes, Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.utils.error import Error


class TensorIterator(Op):
    """
    Loop layer that iterates over tensors and execute embedded sub-graph.
    """

    op = 'TensorIterator'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'input_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'output_port_map': [],  # a list of dicts with such attrs as external_port_id, etc.
            'back_edges': [],  # a list of dicts with such attrs as from_layer, from_port, etc.
            'body': None,  # an Graph object with a body sub-graph
            'sub_graphs': ['body'],  # built-in attribute with all sub-graph
            'infer': self.infer,
            'type_infer': self.ti_type_infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def cover_body_input_data_nodes_with_parameter_ops(ti: Node):
        body = ti.body

        op_port_map = []
        for record in ti.input_port_map:
            operation_node = get_internal_node_by_layer_id(ti, record['internal_layer_id'])
            real_in_port = TensorIterator.special_port_to_real_port(operation_node, copy(record['internal_port_id']))
            op_port_map.append((operation_node, real_in_port))

        for operation_node, in_port in op_port_map:
            data_node = operation_node.in_node(in_port)

            attrs = deepcopy(body.get_edge_data(data_node.id, operation_node.id)[0])
            body.remove_edge(data_node.id, operation_node.id)

            assert data_node.has_valid('shape'), \
                'Data node should have `shape` attribute set, but it`s not for node {}'.format(data_node.id)
            shape = data_node['shape'].copy()
            parameter_data_node = Parameter(body, {'shape': shape_array(shape)}).create_node_with_data()

            body.create_edge(src_node=parameter_data_node, dst_node=operation_node,
                             out_port=0, in_port=in_port, edge_attrs=attrs)
            del body.get_edge_data(parameter_data_node.id, operation_node.id)[0]['out']

    @staticmethod
    def cover_body_constant_data_nodes_with_const_ops(ti: Node):
        body = ti.body
        for data_node in body.get_data_nodes():
            if len(data_node.in_nodes()) == 0 and len(data_node.out_nodes()) != 0:
                assert data_node.has_valid('shape'), \
                    'Data node should have `shape` attribute set, but it`s not for node {}'.format(data_node.id)
                assert data_node.has_valid('value'), \
                    'Data node should have `value` attribute set, but it`s not for node {}'.format(data_node.id)
                shape = data_node['shape'].copy()
                value = data_node['value'].copy()
                const_node = Const(body, {'shape': shape, 'value': value}).create_node()
                body.create_edge(src_node=const_node, dst_node=data_node, out_port=0, in_port=0)

    @staticmethod
    def special_port_to_real_port(node: Node, special_port_id: int, direction: str = 'in'):
        assert node.kind == 'op'
        assert direction in ['in', 'out']

        port_type = 'external_port_id' if node.has_valid('body') else 'internal_port_id'

        if direction == 'in':
            edges = node.in_edges()
        else:
            edges = node.out_edges()

        suitable_edges = {}
        for idx, attrs in edges.items():
            if port_type in attrs and attrs[port_type] == special_port_id:
                suitable_edges[idx] = attrs
        assert len(suitable_edges) == 1
        return list(suitable_edges.keys())[0]

    @staticmethod
    def set_internal_layer_id_for_nodes(ti: Node, nodes: list):
        max_internal_layer_id_used = max([n.soft_get('internal_layer_id', 0) for n in ti.body.get_op_nodes()])

        for node in nodes:
            if not node.has_valid('internal_layer_id'):
                node['internal_layer_id'] = max_internal_layer_id_used = max_internal_layer_id_used + 1

    @staticmethod
    def update_back_edge_map(ti, direction, old_layer_id, old_port_id, new_layer_id, new_port_id=None):
        assert direction in ['from', 'to']
        layer_attr_name = direction + '_layer'
        port_attr_name = direction + '_port'

        for record in ti.back_edges:
            if record[layer_attr_name] != old_layer_id:
                continue
            if (port_attr_name in record and record[port_attr_name] == old_port_id) or new_port_id is not None:
                record[layer_attr_name] = new_layer_id
                if new_port_id is None:
                    del record[port_attr_name]
                else:
                    record[port_attr_name] = new_port_id

    @staticmethod
    def validate_maps(ti):
        def check_by_attribute(port_map, appropriate_attribute, inappropriate_attribute, node_type):
            for record in port_map:
                node = get_internal_node_by_layer_id(ti, record[appropriate_attribute])
                assert node.soft_get('type') == node_type
                assert inappropriate_attribute not in record, record[inappropriate_attribute]

        check_by_attribute(ti.input_port_map, 'internal_layer_id', 'internal_port_id', 'Parameter')
        check_by_attribute(ti.output_port_map, 'internal_layer_id', 'internal_port_id', 'Result')
        check_by_attribute(ti.back_edges, 'from_layer', 'from_port', 'Result')
        check_by_attribute(ti.back_edges, 'to_layer', 'to_port', 'Parameter')

    @staticmethod
    def normalize_internal_ids(ti):
        assert ti.has_valid('input_port_map')
        assert ti.has_valid('output_port_map')
        assert ti.has_valid('back_edges')

        body = ti.body

        TensorIterator.set_internal_layer_id_for_nodes(ti, body.get_op_nodes(type='Parameter'))
        TensorIterator.set_internal_layer_id_for_nodes(ti, body.get_op_nodes(type='Result'))

        node_map = {copy(node.internal_layer_id): node for node in body.get_op_nodes() if
                    node.has_valid('internal_layer_id')}

        for record in ti.input_port_map:
            assert 'internal_layer_id' in record
            assert 'internal_port_id' in record
            assert 'external_port_id' in record

            internal_node_id = copy(record['internal_layer_id'])
            assert internal_node_id in node_map
            internal_node = node_map[internal_node_id]

            in_port = TensorIterator.special_port_to_real_port(internal_node, copy(record['internal_port_id']))
            assert in_port in internal_node.in_ports() and not internal_node.in_port(in_port).disconnected()

            internal_input_node = internal_node.in_port(in_port).get_source().node
            assert internal_input_node.soft_get('type') == 'Parameter'

            TensorIterator.update_back_edge_map(ti=ti, direction='to', old_layer_id=internal_node_id,
                                                old_port_id=record['internal_port_id'],
                                                new_layer_id=internal_input_node.internal_layer_id)
            del record['internal_port_id']
            record['internal_layer_id'] = internal_input_node['internal_layer_id']

        for record in ti.output_port_map:
            assert 'internal_layer_id' in record
            assert 'internal_port_id' in record
            assert 'external_port_id' in record

            internal_node_id = copy(record['internal_layer_id'])
            assert internal_node_id in node_map
            internal_node = node_map[internal_node_id]

            out_port = TensorIterator.special_port_to_real_port(internal_node, copy(record['internal_port_id']), 'out')
            assert out_port in internal_node.out_ports() and not internal_node.out_port(out_port).disconnected()

            assert len(internal_node.out_port(out_port).get_destinations()) >= 1

            internal_output_node = None
            for dst in internal_node.out_port(out_port).get_destinations():
                possible_output_node = dst.node
                if possible_output_node.soft_get('type') == 'Result':
                    assert internal_output_node is None, 'Several Result operations on the same output port of {}'.format(
                        internal_node)
                    internal_output_node = possible_output_node
            assert internal_output_node is not None
            TensorIterator.update_back_edge_map(ti=ti, direction='from', old_layer_id=internal_node_id,
                                                old_port_id=record['internal_port_id'],
                                                new_layer_id=internal_output_node.internal_layer_id)

            del record['internal_port_id']
            record['internal_layer_id'] = internal_output_node.internal_layer_id

        for record in ti.back_edges:
            assert 'from_layer' in record
            assert 'to_layer' in record

            internal_node_id = record['from_layer']
            assert internal_node_id in node_map
            internal_node = node_map[internal_node_id]

            if internal_node.soft_get('type') != 'Result':
                # this output won't get out of the body, but it is still Result and needed on non first iterations of TI
                assert 'from_port' in record
                out_port = TensorIterator.special_port_to_real_port(internal_node, record['from_port'], 'out')
                assert out_port in internal_node.out_ports() and not internal_node.out_port(out_port).disconnected()
                assert len(internal_node.out_port(out_port).get_destinations()) >= 1

                internal_output_node = None
                for dst in internal_node.out_port(out_port).get_destinations():
                    possible_output_node = dst.node
                    if possible_output_node.soft_get('type') == 'Result':
                        assert internal_output_node is None, 'Several Result operations on the same output port of {}' \
                                                             ''.format(internal_node)
                        internal_output_node = possible_output_node
                assert internal_output_node is not None
                TensorIterator.update_back_edge_map(ti=ti, direction='from', old_layer_id=internal_node_id,
                                                    old_port_id=record['from_port'],
                                                    new_layer_id=internal_output_node.internal_layer_id)

        TensorIterator.validate_maps(ti)

    def port_map_attrs(self):
        return [
            'external_port_id',
            'internal_layer_id',
            'internal_port_id',
            'axis',
            'start',
            'stride',
            'end',
            'part_size',
        ]

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
                    ('data', self.backend_attrs() + self.default_backend_attrs, []),
                    '@ports',
                    ('port_map', [], [
                        ('@list', lambda node: self.generate_port_map(node, node.input_port_map, 'in'),
                         ('input', port_map_attrs, [])),
                        ('@list', lambda node: self.generate_port_map(node, node.output_port_map, 'out'),
                         ('output', port_map_attrs, [])),
                    ]),
                    ('back_edges', [], [
                        ('@list', lambda node: self.generate_back_edges(node), ('edge', back_edges_attrs, [])),
                    ]),
                    ('body', [], [('@network', 'body')]),
                ])]
        })

    @staticmethod
    def find_port_id(node: Node, virtual_id: str, attr: str):
        attrs = node.edge({attr: virtual_id})[2]
        assert bool('in' in attrs) != bool('out' in attrs), attrs
        return attrs['in' if 'in' in attrs else 'out']

    @staticmethod
    def find_internal_layer_id(graph: Graph, virtual_id):
        internal_nodes = list(
            filter(lambda d: dict_includes(d[1], {'internal_layer_id': virtual_id}), graph.nodes(data=True)))
        assert len(internal_nodes) == 1, 'Nodes: {}, virtual_id: {}'.format(internal_nodes, virtual_id)
        return internal_nodes[0][0]

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
            result['external_port_id'] = __class__.find_port_id(node, result['external_port_id'], 'external_port_id')
            result['internal_layer_id'] = __class__.find_internal_layer_id(node.body, result['internal_layer_id'])
            result_list.append(result)
        return result_list

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
    def infer(node: Node):
        return
        raise Error('TensorIterator.infer is not implemented. '
                    'Do not insert TensorIterator before middle-end in Model Optimizer')

    @staticmethod
    def ti_type_infer(node):
        from openvino.tools.mo.middle.passes.infer import type_infer
        ti_graph = node.body

        for record in node.input_port_map:
            internal_node = get_internal_node_by_layer_id(node, record['internal_layer_id'])
            assert internal_node.soft_get('type') == 'Parameter', internal_node.soft_get('type')

            real_external_port_idx = TensorIterator.special_port_to_real_port(node, record['external_port_id'])
            external_data_type = node.in_port(real_external_port_idx).get_connection().get_source().get_data_type()
            internal_node.data_type = external_data_type

        fake_input_const_nodes = []
        # create fake const node to make type inference work correctly for all TI input nodes
        for data_node in ti_graph.get_data_nodes(has_value=True):
            if len(data_node.in_nodes()) == 0:
                const_node = Const(ti_graph, {'name': 'const_', 'value': data_node.value}).create_node()
                fake_input_const_nodes.append(const_node)
                ti_graph.create_edge(const_node, data_node)

        type_infer(ti_graph)

        # propagate data types to the TI output ports
        for record in node.output_port_map:
            internal_node = get_internal_node_by_layer_id(node, record['internal_layer_id'])
            assert internal_node.soft_get('type') == 'Result', internal_node.soft_get('type')

            internal_data_type = internal_node.in_port(0).get_data_type()
            real_external_port_idx = TensorIterator.special_port_to_real_port(node, record['external_port_id'], 'out')
            node.out_port(real_external_port_idx).set_data_type(internal_data_type)

        ti_graph.remove_nodes_from([node.id for node in fake_input_const_nodes])

    @staticmethod
    def find_iterations_count_for_output(ti_node):
        def check_field(record, field):
            return field in record and record[field] is not None
        iterations_count = dynamic_dimension_value
        # find out iterations count from inputs.
        # If no input contains 'axis' attribute then no slicing is in TI and it has only one iteration
        # If several inputs have axis attribute with different iterations count then we use maximum value.
        for in_rec in ti_node.input_port_map:
            if not check_field(in_rec, 'axis'):
                continue
            assert check_field(in_rec, 'external_port_id'), "external_port_id not set for input of {} node".format(ti_node.id)
            in_shape = ti_node.in_port(in_rec['external_port_id']).data.get_shape()
            if check_field(in_rec, 'end') and in_rec['end'] >= 0 and \
                    check_field(in_rec, 'start') and in_rec['start'] >= 0:
                in_rec_end = in_rec['end']
                in_rec_start = in_rec['start']
            elif check_field(in_rec, 'end') and in_rec['end'] >= 0:
                in_rec_end = in_rec['end']
                in_rec_start = in_shape[in_rec['axis']] if not check_field(in_rec, 'start') else \
                    in_shape[in_rec['axis']] + 1 + in_rec['start']
            elif check_field(in_rec, 'start') and in_rec['start'] >= 0:
                in_rec_end = in_shape[in_rec['axis']] if not check_field(in_rec, 'end') else \
                    in_shape[in_rec['axis']] + 1 + in_rec['end']
                in_rec_start = in_rec['start']
            elif check_field(in_rec, 'end') and in_rec['end'] < 0 and \
                    check_field(in_rec, 'start') and in_rec['start'] < 0:
                in_rec_end = in_rec['end']
                in_rec_start = in_rec['start']
            else:
                in_rec_end = ti_node.in_port(in_rec['external_port_id']).data.get_shape()[in_rec['axis']]
                in_rec_start = 0

            if check_field(in_rec, 'stride'):
                in_rec_stride = in_rec['stride']
            else:
                in_rec_stride = 1

            # in case of dynamic iterations count don't continue any calculations on this iteration
            if not is_fully_defined(in_rec_end) or not is_fully_defined(in_rec_start):
                continue

            if iterations_count is not dynamic_dimension_value and \
                    ceil((in_rec_end - in_rec_start) / in_rec_stride) != iterations_count:
                raise Error("TensorIterator node {} have inputs with different iterations count".format(ti_node.id))
            iterations_count = ceil((in_rec_end - in_rec_start) / in_rec_stride)

        return iterations_count


def get_internal_node_by_layer_id(ti, internal_layer_id):
    suitable_nodes = ti.body.get_op_nodes(internal_layer_id=internal_layer_id)
    assert len(suitable_nodes) == 1, \
        'Expected 1 node with `internal_layer_id`={}, {} found'.format(internal_layer_id, len(suitable_nodes))
    return suitable_nodes[0]


# Some utils for TI
def _get_internal_idxs_to_names_dict(graph: Graph, ports_type='in'):
    """
    Create mapping from (internal_layer_id, internal_port_id) to layer id in body of TensorIterator.
    """
    mapping = {}
    ordered_nodes = graph.pseudo_topological_sort()
    for node in ordered_nodes:
        if node.kind == 'op' and node.has_valid('internal_layer_id'):
            mapping[node.internal_layer_id] = node.id
    return mapping


def _get_internal_output_node_id(graph: Graph, ti_node_id: str, external_port: int):
    node = Node(graph, ti_node_id)
    outputs = node['output_port_map']
    mapping = _get_internal_idxs_to_names_dict(node['body'], 'out')
    for out in outputs:
        if out['external_port_id'] == external_port:
            return mapping[out['internal_layer_id']]


def _get_internal_input_node_id(graph: Graph, ti_node_id: str, external_port: int):
    node = Node(graph, ti_node_id)
    inputs = node['input_port_map']
    mapping = _get_internal_idxs_to_names_dict(node['body'], 'in')
    for inp in inputs:
        if inp['external_port_id'] == external_port:
            return mapping[inp['internal_layer_id']]
