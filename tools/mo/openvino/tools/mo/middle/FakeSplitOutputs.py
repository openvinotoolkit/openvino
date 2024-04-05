# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.TensorIteratorMerge import TensorIteratorMerge
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.op import Op


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
            Op.normalize_outputs(split_node)


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

        Op.normalize_outputs(node)
