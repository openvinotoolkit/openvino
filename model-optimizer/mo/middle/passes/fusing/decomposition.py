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

import logging as log

import networkx as nx
import numpy as np

from mo.graph.graph import Node
from mo.middle.passes.eliminate import merge_data_nodes
from mo.middle.pattern_match import apply_pattern
from mo.ops.lin_op import Mul, Add
from mo.ops.op import Op


def convert_batch_norm(graph: nx.MultiDiGraph):
    """
    This function finds FusedBatchNorm layer (or BatchNorm for MXNet) and replaces with Mul->Add->Mul->Add sequence.
    """
    for n in list(graph.nodes()):
        node = Node(graph, n)
        if node.has_valid('op') and (
                node.op == 'FusedBatchNorm' or node.op == 'BatchNorm' or node.op == 'BatchNormalization'):
            toutput = node.out_node()
            tinput = node.in_node(0)

            if any([node.in_node(i).value is None for i in range(1, len(node.in_nodes()))]):
                log.warning('Cannot translate FusedBatchNorm {} node with non-constant weights'.format(
                    node.name if node.has_valid('name') else '<UNKNOWN>'))
                continue

            const = node.in_node(1)
            beta = node.in_node(2)
            mean = node.in_node(3)
            variance = node.in_node(4)
            eps = node.eps

            if node.has_valid('fix_gamma') and node.fix_gamma:
                const.value.fill(1.)

            can_be_fused = False if not node.soft_get('can_be_fused') else True

            # Remove edges from FusedBN node
            graph.remove_edge(tinput.id, node.id)
            graph.remove_edge(beta.id, node.id)
            graph.remove_edge(const.id, node.id)
            graph.remove_edge(mean.id, node.id)
            graph.remove_edge(variance.id, node.id)
            graph.remove_edge(node.id, toutput.id)

            scale = 1. / np.sqrt(variance.value + eps)
            shift = (mean.value * (-1)) * scale

            # Expand dims for current layout
            broadcast_dims_cnt = len(tinput.shape) - 2 if graph.graph['layout'] == 'NCHW' else 0
            # Update values and shapes with new shape
            Op.expand_node_shape(const, broadcast_dims_cnt)
            Op.expand_node_shape(beta, broadcast_dims_cnt)

            for idx in range(broadcast_dims_cnt):
                scale = np.expand_dims(scale, axis=-1)
                shift = np.expand_dims(shift, axis=-1)

            _fused_batch_norm_decomposition(graph, tinput, toutput, const, beta, scale, shift, can_be_fused)


def _fused_batch_norm_decomposition(graph: nx.MultiDiGraph, tinput: Node, toutput: Node, gamma: Node, beta: Node,
                                    mean: np.ndarray, variance: np.ndarray, can_be_fused=True):
    """
    This is common function for TF, Caffe and MXNet
    It creates Mul->Add->Mul->Add subgraph
    """
    shape = tinput.shape

    # Create first Mul & Add operations
    mul1_node = Mul(graph, dict(name="Mul1_", can_be_fused=can_be_fused))
    add1_node = Add(graph, dict(name="Add1_", can_be_fused=can_be_fused))

    mul1_data = Op.create_input_data_node(graph, "data_mul_", np.array(mean))
    add1_data = Op.create_input_data_node(graph, "data_add_", np.array(variance))

    # Broadcast const from scalar
    # We can broadcast only when const.value is scalar
    if gamma.shape[0] != gamma.value.shape[0]:
        gamma.value.resize(gamma.shape)
        gamma.value.fill(gamma.value[0])

    # Create second Mul & Add
    mul2_node = Mul(graph, dict(name="Mul2_", can_be_fused=can_be_fused))
    add2_node = Add(graph, dict(name="Add2_", can_be_fused=can_be_fused))

    add2_node.create_node_with_data(
        inputs=[mul2_node.create_node_with_data(
            inputs=[add1_node.create_node_with_data(
                inputs=[mul1_node.create_node_with_data(inputs=[tinput, mul1_data]),
                        add1_data]),
                gamma]),
            beta],
        data_nodes=toutput)


def convert_scale_shift_to_mul_add(graph: nx.MultiDiGraph):
    nodes = [Node(graph, node) for node in graph.nodes() if Node(graph, node).soft_get('op') == 'ScaleShift']
    for node in nodes:
        if node.soft_get('can_be_fused') == False:
            continue

        has_biases = True
        has_weights = True
        # We don't need zero biases
        if len(node.in_nodes()) < 3 or all([x == 0 for x in node.in_node(2).value]):
            has_biases = False
        input_node = node.in_node(0)
        scale_node = node.in_node(1)
        shift_node = node.in_node(2) if has_biases else None
        output_node = node.out_node()

        if all([x == 1 for x in scale_node.value]):
            has_weights = False

        mul_node = Mul(graph, dict(name=node.name + "/Mul_"))
        add_node = Add(graph, dict(name=node.name + "/Add_"))

        # Disconnect ScaleShift node
        graph.remove_edge(input_node.id, node.id)
        graph.remove_edge(node.id, output_node.id)

        # Expand dims for current layout
        broadcast_dims_cnt = len(input_node.shape) - 2 if graph.graph['layout'] == 'NCHW' else 0
        Op.expand_node_shape(scale_node, broadcast_dims_cnt)
        Op.expand_node_shape(shift_node, broadcast_dims_cnt)

        # Connect input->mul->out->add->out
        if has_biases:
            add_node.create_node_with_data(
                inputs=[mul_node.create_node_with_data(inputs=[input_node, scale_node]), shift_node],
                data_nodes=output_node)
        elif has_weights:
            mul_node.create_node_with_data(inputs=[input_node, scale_node], data_nodes=output_node)
        else:
            merge_data_nodes(graph, input_node, output_node)
            graph.remove_node(output_node.id)


def _bn_to_mul_add_action(graph: nx.MultiDiGraph, match: dict):
    # Data nodes
    tinput = match['input']
    toutput = match['output']
    mean = match['mean']
    variance = match['variance']

    # Op node
    bn_node = match['batch_norm']

    # Disconnect data nodes from
    graph.remove_edge(tinput.node, bn_node.node)
    graph.remove_edge(mean.node, bn_node.node)
    graph.remove_edge(variance.node, bn_node.node)

    graph.remove_edge(bn_node.node, toutput.node)

    scale = 1. / np.sqrt(variance.value + bn_node.epsilon)
    shift = (mean.value * (-1)) * scale

    mean.value = np.array(scale)
    variance.value = np.array(shift)

    # Expand dims for current layout
    broadcast_dims_cnt = len(tinput.shape) - 2 if graph.graph['layout'] == 'NCHW' else 0
    # Update values and shapes with new shape
    Op.expand_node_shape(mean, broadcast_dims_cnt)
    Op.expand_node_shape(variance, broadcast_dims_cnt)

    can_be_fused = False if not bn_node.soft_get('can_be_fused') else True

    mul_node = Mul(graph, dict(name="Mul_", can_be_fused=can_be_fused))
    add_node = Add(graph, dict(name="Add_", can_be_fused=can_be_fused))

    # Connect input->mul->add
    add_node.create_node_with_data(inputs=[mul_node.create_node_with_data(inputs=[tinput, mean]), variance],
                                   data_nodes=toutput)


def convert_bn_to_mul_add(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('mean', dict(kind='data')),
            ('variance', dict(kind='data')),
            ('output', dict(kind='data')),
            ('batch_norm', dict(kind='op', op='BatchNormalization')),
        ],
        edges=[
            ('input', 'batch_norm', {'in': 0}),
            ('mean', 'batch_norm', {'in': 1}),
            ('variance', 'batch_norm', {'in': 2}),
            ('batch_norm', 'output'),
        ],
        action=_bn_to_mul_add_action
    )
