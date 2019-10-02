"""
 Copyright (c) 2018-2019 Intel Corporation

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

from collections import deque
from copy import deepcopy

import numpy as np

from extensions.ops.tensor_iterator import TensorIterator
from mo.graph.graph import Node, Graph, add_opoutput
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.op import Op
from mo.ops.reshape import Reshape
from mo.utils.graph import sub_graph_between_nodes, invert_sub_graph_between_nodes

stop_nodes = ['TensorIteratorInput', 'TensorIteratorOutput', 'TensorIteratorBackEdge', 'TensorIteratorCondition']


def op_type(graph, node_name: str):
    node = Node(graph, node_name)
    if node.has_valid('kind') and node['kind'] == 'op':
        return node['op']
    else:
        return None


def update_inputs(graph, inputs: list, node_name: str):
    node = Node(graph, node_name)
    if node.has_valid('kind') and node['kind'] == 'op' and node['op'] == 'TensorIteratorInput':
        if node_name not in inputs:
            inputs.append(node_name)


def reverse_dfs(graph: Graph, node_name: str, stop_nodes: list, inputs: list, visited: set = None):
    d = deque()

    if visited is None:
        visited = set()
    visited.add(node_name)
    d.appendleft(node_name)
    while len(d) != 0:
        cur_node = d.popleft()
        for in_node_name, _ in graph.in_edges(cur_node):
            if in_node_name not in visited:
                if op_type(graph, in_node_name) not in stop_nodes:
                    visited.add(in_node_name)
                    d.append(in_node_name)
                else:
                    update_inputs(graph, inputs, in_node_name)


def dfs(graph: Graph, node_name: str, stop_nodes: list, visited: set = None):
    d = deque()

    visited.add(node_name)
    d.appendleft(node_name)
    while len(d) != 0:
        cur_node = d.popleft()
        for _, out_node_name in graph.out_edges(cur_node):
            if out_node_name not in visited:
                if op_type(graph, out_node_name) not in stop_nodes:
                    visited.add(out_node_name)
                    d.append(out_node_name)


def get_body(graph, inputs, outputs):
    if len(inputs) == 0:
        nodes, extra_inputs = invert_sub_graph_between_nodes(
            graph,
            outputs,
            inputs,
            lambda node: node.soft_get('op') == 'TensorIteratorInput'
        )
    else:
        nodes, extra_inputs = sub_graph_between_nodes(
            graph,
            inputs,
            outputs,
            lambda node: node.soft_get('op') == 'TensorIteratorInput'
        )
    nodes = list(set(nodes) - set(inputs) - set(outputs) - set(extra_inputs))
    return nodes, extra_inputs


class TensorIteratorMerge(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['is_cyclic']]

    def run_after(self):
        return []

    def run_before(self):
        return []

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('condition', dict(kind='op', op='TensorIteratorCondition')),
            ],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph, match: dict):
        # Here we will found all parts of TI: condition, inputs/outputs, back edges, body and create TensorIterator Op
        # and make all checks needed for TensorIteator work
        cond_data = match['condition'].out_node(0)
        time_data = match['condition'].out_node(1) if len(match['condition'].out_nodes()) > 1 else None
        name = match['condition'].name

        back_edges = []
        inputs = []
        outputs = []

        for node in cond_data.out_nodes():
            if node['kind'] == 'op' and node['op'] == 'TensorIteratorBackEdge':
                back_edges.append(node.id)
            elif node['kind'] == 'op' and node['op'] == 'TensorIteratorInput':
                inputs.append(node.id)
            elif node['kind'] == 'op' and node['op'] == 'TensorIteratorOutput':
                outputs.append(node.id)

        if time_data is not None:
            for node in time_data.out_nodes():
                if node['kind'] == 'op' and node['op'] == 'TensorIteratorInput':
                    inputs.append(node.id)
                elif node['kind'] == 'op' and node['op'] == 'TensorIteratorOutput':
                    outputs.append(node.id)
                else:
                    # something goes wrong here
                    assert False
        condition = match['condition']
        tensor_sequence_length = condition.in_node(0)
        graph.remove_nodes_from([condition.id, cond_data.id, tensor_sequence_length.id])
        if time_data is not None:
            graph.remove_nodes_from([time_data.id])

        body_nodes, extra_inputs = get_body(graph, inputs, outputs)
        body_nodes = list(set(body_nodes) - set([cond_data]))

        inputs += extra_inputs

        assert all([node in graph.nodes() for node in body_nodes])

        inputs = [Node(graph, node) for node in inputs]
        outputs = [Node(graph, node) for node in outputs]
        back_edges = [Node(graph, node) for node in back_edges]

        external_inputs = [
            {
                'external_data_id': node.in_node(1 if node.has_valid('axis') else 0),
                'internal_data_id': node.out_node(0),
                'axis': node.axis,
                'start': node.start,
                'end': node.end,
                'stride': node.stride,
                'part_size': node.part_size
            } for node in inputs]

        external_outputs = [
            {
                'external_data_id': node.out_node(0),
                'internal_data_id': node.in_node(1 if node.has_valid('axis') else 0),
                'axis': node.axis,
                'start': node.start,
                'end': node.end,
                'stride': node.stride,
                'part_size': node.part_size
            } for node in outputs]

        back_edges_data = [
            {
                'from_data_id': node.in_node(1),
                'to_data_id': node.out_node(0),
                'init_data_id': node.in_node(0),
            } for node in back_edges
        ]

        body = Graph(name='body')
        body.graph = graph.graph
        body.add_nodes_from([(node, graph.node[node]) for node in body_nodes])
        body.add_edges_from(
            [(u, v, k, d) for u, v, k, d in graph.edges(data=True, keys=True) if u in body_nodes and v in body_nodes])

        graph.remove_nodes_from(
            body_nodes + [match['condition'].id] + [inp.id for inp in inputs] + [out.id for out in outputs])
        internal_id_count = 0
        real_back_edges = []
        for edge in back_edges_data:
            assert edge['from_data_id'].id in body.nodes()
            assert edge['to_data_id'].id in body.nodes()
            assert edge['init_data_id'].id in body.nodes()
            edge['from_data_id'] = Node(body, edge['from_data_id'].id)
            edge['to_data_id'] = Node(body, edge['to_data_id'].id)
            edge['init_data_id'] = Node(body, edge['init_data_id'].id)
            add_opoutput(body, edge['from_data_id'].id, 0, False)

            # Assign/reuse ids for the back-edge start; it comes from from_data_id
            assert len(edge['from_data_id'].in_nodes()) == 1
            # layer id
            if not edge['from_data_id'].in_node().has_valid('internal_layer_id'):
                edge['from_data_id'].in_node()['internal_layer_id'] = internal_id_count
                internal_id_count += 1
            edge['from_layer'] = edge['from_data_id'].in_node()['internal_layer_id']

            # port id
            if 'internal_port_id' not in edge['from_data_id'].in_edge():
                edge['from_data_id'].in_edge()['internal_port_id'] = internal_id_count
                internal_id_count += 1
            edge['from_port'] = edge['from_data_id'].in_edge()['internal_port_id']

            # Look at all consumers for a data that ends a back-edge
            # For each such consumer, there will be a separate back-edge (and input)
            current_real_back_edges = []
            for _, consumer, key, edge_attrs in body.out_edges(edge['to_data_id'].id, data=True, keys=True):

                real_edge = {}
                real_edge.update(edge)  # all real back_edges have the same back-edge start

                consumer = Node(body, consumer)

                if real_edge['to_data_id'].in_node().has_valid('internal_layer_id'):
                    assert False
                    real_edge['to_data_id'].out_node()['internal_layer_id'] = \
                        real_edge['to_data_id'].in_node().internal_layer_id
                elif not consumer.has_valid('internal_layer_id'):
                    consumer['internal_layer_id'] = internal_id_count
                    internal_id_count += 1
                real_edge['to_layer'] = consumer['internal_layer_id']

                assert 'internal_port_id' not in edge_attrs
                assert len(real_edge['init_data_id'].out_edges()) == 1
                assert not 'internal_port_id' in real_edge['init_data_id'].out_edge()
                edge_attrs['internal_port_id'] = internal_id_count
                internal_id_count += 1
                real_edge['to_port'] = edge_attrs['internal_port_id']
                real_edge['consumer'] = consumer
                real_edge['consumer_key'] = key

                real_edge['attrs'] = deepcopy(edge_attrs)
                current_real_back_edges.append(real_edge)

            # connect initial data node with each consumer providing actual edge attributes
            body.add_edges_from([
                (
                    real_edge['init_data_id'].id,
                    real_edge['consumer'].id,
                    real_edge['consumer_key'],
                    real_edge['attrs'])
                for real_edge in current_real_back_edges])

            body.remove_nodes_from([edge['to_data_id'].id, edge['to_data_id'].in_node().id])
            real_back_edges += current_real_back_edges

        real_external_inputs = []

        for ext_inp in external_inputs:
            assert ext_inp['external_data_id'].id not in body.nodes()
            assert ext_inp['internal_data_id'].id in body.nodes()
            ext_inp['internal_data_id'] = Node(body, ext_inp['internal_data_id'].id)

            if ext_inp['axis'] is not None:
                # Insert squeezing resize at input port that has partitioning
                shape = ext_inp['internal_data_id'].shape.copy()
                assert not ext_inp['internal_data_id'].has_valid('value')
                new_input_data = Op._create_data_node(body, ext_inp['internal_data_id'].name + '/UnsqueezedInput',
                                                      dict(shape=np.insert(shape, ext_inp['axis'], 1)))
                dim = shape.copy()
                # try to do it dynamically reshapable along one of the axis
                # it is practically useful to reshape along batch dimension, but here we cannot detect where it is
                # so, we are guessing based on other transformations that it is the major dimension
                dim[0] = -1
                reshape_op = Reshape(body, dict(name=ext_inp['internal_data_id'].name + '/InputSqueeze'))
                reshape_dim_data = Const(body, {'name': ext_inp['internal_data_id'].name + '/ReshapeDim',
                                                'value': dim}).create_node_with_data()
                reshape_op.create_node_with_data([new_input_data, reshape_dim_data],
                                                 data_nodes=[ext_inp['internal_data_id']])
                ext_inp['internal_data_id'] = new_input_data

            ext_inp['internal_data_id']['is_input'] = True
            assert len(ext_inp['internal_data_id'].in_nodes()) == 0
            ext_inp['external_port_id'] = internal_id_count
            internal_id_count += 1
            for _, consumer, edge_attrs in body.out_edges(ext_inp['internal_data_id'].id, data=True):
                real_ext_inp = {}
                real_ext_inp.update(ext_inp)
                consumer = Node(body, consumer)
                if not consumer.has_valid('internal_layer_id'):
                    consumer['internal_layer_id'] = internal_id_count
                    internal_id_count += 1
                if not 'internal_port_id' in edge_attrs:
                    edge_attrs['internal_port_id'] = internal_id_count
                    internal_id_count += 1
                real_ext_inp['internal_layer_id'] = consumer['internal_layer_id']
                real_ext_inp['internal_port_id'] = edge_attrs['internal_port_id']
                real_external_inputs.append(real_ext_inp)

        for ext_out in external_outputs:
            assert ext_out['external_data_id'].id not in body.nodes()
            assert ext_out['internal_data_id'].id in body.nodes()
            ext_out['internal_data_id'] = Node(body, ext_out['internal_data_id'].id)

            if ext_out['axis'] is not None:
                # Insert unsqueezing resize at output port that has partitioning
                dim = ext_out['internal_data_id'].shape.copy()
                # trying to make it dynamically reshapable (see related comment above for the first Reshape)
                dim[0] = -1
                assert not ext_out['internal_data_id'].has_valid('value')
                reshape_op = Reshape(body, dict(name=ext_out['internal_data_id'].name + '/OutputUnsqueeze'))
                reshape_dim_data = Const(body, {'name': ext_out['internal_data_id'].name + '/ReshapeDim',
                                                'value': np.insert(dim, ext_out['axis'], 1)}).create_node_with_data()
                ext_out['internal_data_id'] = reshape_op.create_node_with_data([ext_out['internal_data_id'],
                                                                                reshape_dim_data])

            # TODO: add here working with simple outputs

            add_opoutput(body, ext_out['internal_data_id'].id, 0, False)
            # assert len(ext_out['internal_data_id'].out_nodes()) == 0
            assert len(ext_out['internal_data_id'].in_nodes()) == 1
            if not 'internal_layer_id' in ext_out['internal_data_id'].in_node():
                ext_out['internal_data_id'].in_node()['internal_layer_id'] = internal_id_count
                internal_id_count += 1
            if not 'internal_port_id' in ext_out['internal_data_id'].in_edge():
                ext_out['internal_data_id'].in_edge()['internal_port_id'] = internal_id_count
                internal_id_count += 1
            ext_out['internal_layer_id'] = ext_out['internal_data_id'].in_node()['internal_layer_id']
            ext_out['internal_port_id'] = ext_out['internal_data_id'].in_edge()['internal_port_id']
            ext_out['external_port_id'] = internal_id_count
            internal_id_count += 1

        ti_op = TensorIterator(graph, {
            'name': name + '/TensorIterator',
            'body': body,
            'in_ports_count': len(external_inputs),
            'out_ports_count': len(external_outputs),

            'input_port_map': [
                {field: external_input[field] for field in
                 ['external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'stride', 'part_size', 'start',
                  'end']}
                for external_input in real_external_inputs],

            'output_port_map': [
                {field: external_output[field] for field in
                 ['external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'stride', 'part_size', 'start',
                  'end']}
                for external_output in external_outputs],
            'back_edges': [
                {field: edge[field] for field in ['from_layer', 'from_port', 'to_layer', 'to_port']}
                for edge in real_back_edges],
        })

        ti_outs = ti_op.create_node_with_data(
            inputs=[inp['external_data_id'] for inp in external_inputs],
            edge_attrs=[{'external_port_id': inp['external_port_id']} for inp in external_inputs],
            data_nodes=[out['external_data_id'] for out in external_outputs]
        )

        if not isinstance(ti_outs, list):
            ti_outs = [ti_outs]

        for i, out in enumerate(ti_outs):
            out.in_edge()['external_port_id'] = external_outputs[i]['external_port_id']
