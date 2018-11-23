"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.graph.graph import Node
from mo.utils.graph import sub_graph_between_nodes
from mo.middle.replacement import MiddleReplacementPattern
from extensions.ops.tensor_iterator import TensorIterator
from mo.ops.op import Op
from mo.ops.reshape import Reshape

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


def reverse_dfs(graph: nx.MultiDiGraph, node_name: str, stop_nodes: list, inputs: list, visited: set = None):
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

def dfs(graph: nx.MultiDiGraph, node_name: str, stop_nodes: list, visited: set = None):
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

def get_body(graph, cond, inputs, outputs):
    nodes, extra_inputs = sub_graph_between_nodes(
        graph,
        [cond] + inputs,
        outputs,
        lambda node: node.soft_get('op')  == 'TensorIteratorInput'
    )
    nodes = list(set(nodes) - set([cond] + inputs) - set(outputs) - set(extra_inputs))
    return nodes, extra_inputs
    #return nx.MultiDiGraph()


class TensorIteratorMerge(MiddleReplacementPattern):
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
        time_data = match['condition'].out_node(1)
        name = match['condition'].name

        assert match['condition'].in_node(0).has_valid('value')
        assert match['condition'].in_node(1).has_valid('value')

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

        for node in time_data.out_nodes():
            if node['kind'] == 'op' and node['op'] == 'TensorIteratorInput':
                inputs.append(node.id)
            elif node['kind'] == 'op' and node['op'] == 'TensorIteratorOutput':
                outputs.append(node.id)
            else:
                # something goes wrong here
                assert False

        graph.remove_nodes_from([cond_data.id, time_data.id])

        body_nodes, extra_inputs = get_body(graph, match['condition'].id, inputs, outputs)
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
                'init_data_id': node.in_node(0)
            } for node in back_edges
        ]
        
        body = nx.MultiDiGraph(name='body')
        body.graph['layout'] = graph.graph['layout']
        body.add_nodes_from([(node, graph.node[node]) for node in body_nodes])
        body.add_edges_from([(u,v,k,d)for u,v,k,d in graph.edges(data=True, keys=True) if u in body_nodes and v in body_nodes])
        
        graph.remove_nodes_from(body_nodes + [match['condition'].id] + [inp.id for inp in inputs] + [out.id for out in outputs])

        for i, edge in enumerate(back_edges_data, start=0):
            assert edge['from_data_id'].id in body.nodes()
            assert edge['to_data_id'].id in body.nodes()
            assert edge['init_data_id'].id in body.nodes()
            edge['from_data_id'] = Node(body, edge['from_data_id'].id)
            edge['to_data_id'] = Node(body, edge['to_data_id'].id)
            edge['init_data_id'] = Node(body, edge['init_data_id'].id)
            edge['from_data_id']['is_output'] = True

            if not edge['from_data_id'].in_node().has_valid('internal_layer_id'):
                edge['from_data_id'].in_node()['internal_layer_id'] = 4*i+0
            edge['from_layer'] = edge['from_data_id'].in_node()['internal_layer_id']
            if 'internal_port_id' not in edge['from_data_id'].in_edge():
                edge['from_data_id'].in_edge()['internal_port_id'] = 4*i+1
            edge['from_port'] = edge['from_data_id'].in_edge()['internal_port_id']

            #assert not edge['to_data_id'].out_node().has_valid('internal_layer_id')
            if edge['to_data_id'].in_node().has_valid('internal_layer_id'):
                edge['to_data_id'].out_node()['internal_layer_id'] = edge['to_data_id'].in_node().internal_layer_id
            elif not edge['to_data_id'].out_node().has_valid('internal_layer_id'):
                edge['to_data_id'].out_node()['internal_layer_id'] = 4*i+2
            edge['to_layer'] = edge['to_data_id'].out_node()['internal_layer_id']
            
            assert 'internal_port_id' not in edge['to_data_id'].out_edge()
            if 'internal_port_id' in edge['init_data_id'].out_edge():
                edge['to_data_id'].out_edge()['internal_port_id'] = edge['init_data'].out_edge()['internal_port_id']
            else:
                edge['to_data_id'].out_edge()['internal_port_id'] = 4*i+3
            edge['to_port'] = edge['to_data_id'].out_edge()['internal_port_id']

            body.add_edges_from([(edge['init_data_id'].id, edge['to_data_id'].out_node().id, deepcopy(edge['to_data_id'].out_edge()))])
            body.remove_nodes_from([edge['to_data_id'].in_node().id, edge['to_data_id'].id])

        for i, ext_inp in enumerate(external_inputs, start=4*len(back_edges_data)):
            assert ext_inp['external_data_id'].id not in body.nodes()
            assert ext_inp['internal_data_id'].id in body.nodes()
            ext_inp['internal_data_id'] = Node(body, ext_inp['internal_data_id'].id)

            if ext_inp['axis'] is not None:
                # Insert squeezing resize at input port that has partitioning
                shape = ext_inp['internal_data_id'].shape.copy()
                assert not ext_inp['internal_data_id'].has_valid('value')
                new_input_data = Op._create_data_node(body, ext_inp['internal_data_id'].name + '/UnsqueezedInput', dict(shape=np.insert(shape, ext_inp['axis'], 1)))
                reshape_op = Reshape(body, dict(name=ext_inp['internal_data_id'].name + '/InputSqueeze', dim=shape))
                reshape_op.create_node_with_data([new_input_data], data_nodes=[ext_inp['internal_data_id']])
                ext_inp['internal_data_id'] = new_input_data

            ext_inp['internal_data_id']['is_input'] = True
            assert len(ext_inp['internal_data_id'].in_nodes()) == 0
            assert len(ext_inp['internal_data_id'].out_nodes()) == 1
            if not 'internal_layer_id' in  ext_inp['internal_data_id'].out_node():
                ext_inp['internal_data_id'].out_node()['internal_layer_id'] = i
            if not 'internal_port_id' in ext_inp['internal_data_id'].out_edge():
                ext_inp['internal_data_id'].out_edge()['internal_port_id'] = i
            ext_inp['internal_layer_id'] = ext_inp['internal_data_id'].out_node()['internal_layer_id']
            ext_inp['internal_port_id'] = ext_inp['internal_data_id'].out_edge()['internal_port_id']
            ext_inp['external_port_id'] = i

        for i, ext_out in enumerate(external_outputs, start=4*len(back_edges_data) + len(external_inputs)):
            assert ext_out['external_data_id'].id not in body.nodes()
            assert ext_out['internal_data_id'].id in body.nodes()
            ext_out['internal_data_id'] = Node(body, ext_out['internal_data_id'].id)

            if ext_out['axis'] is not None:
                # Insert unsqueezing resize at output port that has partitioning
                shape = ext_out['internal_data_id'].shape.copy()
                assert not ext_out['internal_data_id'].has_valid('value')
                reshape_op = Reshape(body, dict(name=ext_out['internal_data_id'].name + '/OutputUnsqueeze', dim=np.insert(shape, ext_out['axis'], 1)))
                ext_out['internal_data_id'] = reshape_op.create_node_with_data([ext_out['internal_data_id']])

            # TODO: add here working with simple outputs

            ext_out['internal_data_id']['is_output'] = True
            #assert len(ext_out['internal_data_id'].out_nodes()) == 0
            assert len(ext_out['internal_data_id'].in_nodes()) == 1
            if not 'internal_layer_id' in ext_out['internal_data_id'].in_node():
                ext_out['internal_data_id'].in_node()['internal_layer_id'] = i
            if not 'internal_port_id' in ext_out['internal_data_id'].in_edge():
                ext_out['internal_data_id'].in_edge()['internal_port_id'] = i
            ext_out['internal_layer_id'] = ext_out['internal_data_id'].in_node()['internal_layer_id']
            ext_out['internal_port_id'] = ext_out['internal_data_id'].in_edge()['internal_port_id']
            ext_out['external_port_id'] = i

        ti_op = TensorIterator(graph, {
            'name': name + '/TensorIterator',
            'body': body,

            # FOR TESTING PURPOSES
            'input_port_map': [
                {field: external_input[field] for field in [ 'external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'stride', 'part_size', 'start', 'end']}
                for external_input in external_inputs],

            'output_port_map': [
                {field: external_output[field] for field in [ 'external_port_id', 'internal_layer_id', 'internal_port_id', 'axis', 'stride', 'part_size', 'start', 'end']}
                for external_output in external_outputs],
            'back_edges': [
                {field: edge[field] for field in [ 'from_layer', 'from_port', 'to_layer', 'to_port']}
                for edge in back_edges_data],
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



        # Create TI operation
