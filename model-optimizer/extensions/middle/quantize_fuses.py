# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.middle.BinarizeWeightsM1P1 import BinarizeWeightsM1P1
from extensions.middle.DeleteControlFlowEdges import DeleteControlFlowEdges
from extensions.middle.EltwiseChecker import EltwiseChecker
from extensions.middle.quantize_linear_resolver import QuantizeLinearResolver
from mo.graph.graph import Graph
from mo.middle.passes.fusing.helpers import get_value_in_port
from mo.middle.replacement import MiddleReplacementPattern


class MarkNodesToFuseUpToFakeQuantize(MiddleReplacementPattern):
    """
        Marks special nodes that could be pulled through Quantize operation.
        Sets `fuse_up_to_quantize_ports` parameter to list of indexes of input ports of Quantize operation
        where specified node should appear.

    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_fusing]

    def run_after(self):
        return [DeleteControlFlowEdges]

    def run_before(self):
        return []

    @staticmethod
    def mark_fusable_muls_on_weights(graph):
        for node in graph.get_op_nodes(op='Mul'):
            children = node.out_port(0).get_destinations()
            if len(children) > 1 or children[0].node.soft_get('type') not in ['Convolution', 'Deconvolution', 'MatMul']:
                continue
            value_in_port = get_value_in_port(node)
            if value_in_port is None:
                continue
            value_shape = value_in_port.data.get_shape()
            non_one_axis = np.argwhere(value_shape != 1)
            if non_one_axis.size != 1:
                continue
            non_one_axis = non_one_axis.item(0)
            node['can_be_fused'] = True
            EltwiseChecker().mark_eltwise_node(node, non_one_axis)

    def find_and_replace_pattern(self, graph: Graph):
        # to prevent fusing of non per channel lin ops, we run EltwiseChecker to mark nodes with can_be_fused attribute
        EltwiseChecker().find_and_replace_pattern(graph)
        self.mark_fusable_muls_on_weights(graph)
        eltwise_nodes = graph.get_op_nodes(op='Mul', can_be_fused=True) + \
                        graph.get_op_nodes(op='Sub', can_be_fused=True) + \
                        graph.get_op_nodes(op='Add', can_be_fused=True)
        for elt in eltwise_nodes:
            if elt.in_port(0).data.get_value() is not None or elt.in_port(1).data.get_value() is not None:
                elt['fuse_up_to_quantize_ports'] = [3, 4]

        slice = graph.get_op_nodes(op='Slice')
        for sl in slice:
            sl['fuse_up_to_quantize_ports'] = [0]


class FakeQuantizeFuse(MiddleReplacementPattern):
    """
        Pulls nodes containing `fuse_up_to_quantize_ports` parameter (node to fuse) through Quantize operation

        If `fuse_up_to_quantize_ports` list contains one input port to which node to fuse should be delivered,
            replacer reconnects edges.

        If `fuse_up_to_quantize_ports` list contains more than one input port to which node to fuse should be delivered,
            replacer reconnects edges of first port from `fuse_up_to_quantize_ports` list, for other ports
            replacer duplicates node to fuse (duplicate connections of inputs of node to fuse to duplicates of it)
    """
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_fusing]

    def run_after(self):
        return [QuantizeLinearResolver]

    def run_before(self):
        return [BinarizeWeightsM1P1]

    def find_and_replace_pattern(self, graph: Graph):
        for quantize_node in graph.get_op_nodes(op='FakeQuantize'):
            while len(quantize_node.out_port(0).get_destinations()) == 1:
                if not quantize_node.out_port(0).get_destination().node.has_valid('fuse_up_to_quantize_ports'):
                    break
                fuse_node = quantize_node.out_port(0).get_destination().node
                quantize_to_mul_in_port_index = quantize_node.out_port(0).get_destination().idx

                # connecting the rest of model after mul to quantize, mul node hangs on quantize
                fuse_node.out_port(0).get_connection().set_source(quantize_node.out_port(0))

                # mul node is disconnected from the graph
                fuse_node.in_port(quantize_to_mul_in_port_index).disconnect()

                first_port_fusion = True
                for in_quantize_port in fuse_node['fuse_up_to_quantize_ports']:
                    fuse_node_duplicate = fuse_node
                    if not first_port_fusion:
                        fuse_node_duplicate = fuse_node.copy_node(
                            {'in_ports_count': len(fuse_node.in_ports()),
                             'out_ports_count': len(fuse_node.out_ports())})

                    quantize_node.in_port(in_quantize_port).get_connection().set_destination(
                        fuse_node_duplicate.in_port(quantize_to_mul_in_port_index))

                    fuse_node_duplicate.out_port(0).connect(quantize_node.in_port(in_quantize_port))

                    if not first_port_fusion:
                        for idx, port in fuse_node.in_ports().items():
                            if idx == quantize_to_mul_in_port_index:
                                continue
                            port.get_source().connect(fuse_node_duplicate.in_port(idx))
                    fuse_node_duplicate.infer(fuse_node_duplicate)

                    first_port_fusion = False