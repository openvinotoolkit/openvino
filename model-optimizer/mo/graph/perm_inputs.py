"""
 Copyright (c) 2019 Intel Corporation

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
import networkx as nx

from extensions.ops.gather import Gather
from mo.graph.graph import Node
from mo.ops.const import Const
from mo.ops.op import PermuteAttrs


def get_node_with_permutation(node: Node, port_info: str):
    node_type, port = port_info.split(':')
    port = int(port)
    return node.in_node(port) if node_type == 'input' else node.out_node(port)


def axis(op_node: Node, port_info: str, input_port: int):
    """
    Performs layout change related transformation of the data on the in_port_idx port of op_node.
    Translates shape indexes from one layout to another according to inverse permutation

    Transformation inserts Gather operation with
        permutation as 0-port input data and
        actual data to translate as 1-port input indexes of Gather

    For example:
        NHWC Reduce operation has 0-port input with data of shape [1, 2, 3, 4] and
        1-port input with axis indices [0, 1].

        After translating such operation to NCHW layout:
            0-port input shape = [1, 4, 2, 3]
            1-port input axis indices = [0, 2]
    """
    graph = op_node.graph

    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    data_node = op_node.in_node(input_port)

    const = Const(graph, {'value': permutation.inv, 'need_shape_inference': True}).create_node_with_data()
    gather = Gather(graph, {'name': op_node.name + '/AxisGather', 'need_shape_inference': True}).create_node_with_data([const, data_node])
    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()
    graph.add_edge(gather.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)
    op_node['need_shape_inference'] = True


def order(op_node: Node, port_info: str, input_port: int):
    """
        Performs layout change related transformation of the data on the in_port_idx port of op_node.
        Translates ordered shape indexes from one layout to another according to permutation

        Transformation inserts two Gather operations

        1 Gather reorders data to new layout according to direct permutation:
            actual data to translate as 1-port input indexes of Gather and
            permutation as 0-port input data
        2 Gather translates shape indexes from one layout to another according to inverse permutation
            permutation as 0-port input data and
            actual data to translate as 1-port input indexes of Gather

    For example:
        NHWC Transpose operation has 0-port input with data of shape [1, 2, 3, 4] and
        1-port input with new order indices [0, 1, 3, 2].

        After translating such operation to NCHW layout:
            0-port input shape = [1, 4, 2, 3]

        1 phase (after first Gather insertion):
            1-port input order indices = [0, 2, 1, 3]
        2 phase (after second Gather insertion):
            1-port input order indices = [0, 3, 2, 1]
    """
    graph = op_node.graph
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    data_node = op_node.in_node(input_port)

    const = Const(graph, {'value': permutation.perm, 'need_shape_inference': True}).create_node_with_data()
    gather = Gather(graph, {'name': op_node.name + '/OrderGather_1',
                            'need_shape_inference': True}).create_node_with_data([data_node, const])

    const_1 = Const(graph, {'value': permutation.inv, 'need_shape_inference': True}).create_node_with_data()
    gather_1 = Gather(graph, {'name': op_node.name + '/OrderGather_2',
                              'need_shape_inference': True}).create_node_with_data([const_1, gather])

    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()
    graph.add_edge(gather_1.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)
    op_node['need_shape_inference'] = True


def shape(op_node: Node, port_info: str, input_port: int):
    graph = op_node.graph
    permutation_data_node = get_node_with_permutation(op_node, port_info)
    assert permutation_data_node.has_and_set('permutation'), 'Data node "{}" does not have permutation for node {}, ' \
                                                             'port_info "{}".'.format(permutation_data_node.id,
                                                                                      op_node.id, port_info)
    permutation = permutation_data_node.permutation
    if len(permutation.perm) == 0:
        return

    data_node = op_node.in_node(input_port)

    const = Const(graph, {'value': permutation.perm, 'need_shape_inference': True}).create_node_with_data()
    gather = Gather(graph, {'name': op_node.name + '/ShapeGather',
                            'need_shape_inference': True}).create_node_with_data([data_node, const])
    attrs = graph.get_edge_data(data_node.id, op_node.id, key=0).copy()

    graph.add_edge(gather.id, op_node.id, **attrs)
    graph.remove_edge(data_node.id, op_node.id)

    # need to run manually to override output shape value to resolve shape collision for nodes with
    # 'correct_data_layout' output port attrs
    op_node.infer(op_node)


class PermuteInputs:
    common_inv_permutation = lambda node, port_info, input_port: axis(node, port_info, input_port)

    input_permutes = {
        'axis': common_inv_permutation,
        'order': lambda node, port_info, input_port: order(node, port_info, input_port),
        'shape': lambda node, port_info, input_port: shape(node, port_info, input_port),
    }

    def set_input_permutation(self, node1: Node, node2: Node, port_info: str, permutation_rule: str):
        """
        Sets input permutation attribute on the edge between node1 and node2.
        Input permutation consists of function that perform input permutation and
        input port info 'input' or 'output' + <port_number> that points on the input with PermuteAttr.Permutation which
        current input depends on
        """
        assert permutation_rule in self.input_permutes, 'No `{}` permutation rule in {}'.format(permutation_rule,
                                                                                                __class__.__name__)
        nx.set_edge_attributes(G=node1.graph,
                               values={(node1.id, node2.id, 0): (self.input_permutes[permutation_rule],
                                                                 port_info)},
                               name='input_permutation')
