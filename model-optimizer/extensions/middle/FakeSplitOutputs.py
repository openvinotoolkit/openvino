# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.middle.TensorIteratorMerge import TensorIteratorMerge
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.result import Result


class AddFakeOutputsToSplit(MiddleReplacementPattern):
    """
        Adding fake outputs for Split nodes in case when it has less output ports than split parts:
        This pass:
            1. Looking for Split operations
            2. Check that Split have less connected output ports than split parts
            3. For every missed port adding this port, Output operation to this port
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge]

    def find_and_replace_pattern(self, graph: Graph):
        for split_node in graph.get_op_nodes(op='Split'):
            AddFakeOutputsToSplit.split_normalize_outputs(split_node)

    @staticmethod
    def split_normalize_outputs(node: Node):
        if node.has_valid('out_ports_count') and len(node.out_edges()) < node.out_ports_count:
            for p in range(node.out_ports_count):
                if p not in node.out_ports():
                    node.add_output_port(p)
                if node.out_port(p).disconnected():
                    res_node = Result(node.graph, {'name': node.name + '/Fake_output_{}/'.format(p),
                                                   'keep_output_port': True}).create_node()
                    node.out_port(p).connect(res_node.in_port(0))


class AddFakeOutputsToVariadicSplit(MiddleReplacementPattern):
    """
        Adding fake outputs for VariadicSplit nodes in case when it has less output ports than split parts:
        This pass:
            1. Looking for VariadicSplit operations
            2. Check that VariadicSplit have less connected output ports than split parts
            3. For every missed port adding this port, Output operation to this port
    """

    enabled = True

    def run_after(self):
        return [TensorIteratorMerge]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(kind='op', op='VariadicSplit'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        axis = node.in_port(1).data.get_value()
        size_splits = node.in_port(2).data.get_value()

        output_shape = sum([node.out_node(port).shape[axis] for port in node.out_nodes()])

        if output_shape == node.in_port(0).data.get_shape()[axis]:
            return

        if not node.has_valid('out_ports_count'):
            node['out_ports_count'] = len(size_splits)

        AddFakeOutputsToSplit().split_normalize_outputs(node)
