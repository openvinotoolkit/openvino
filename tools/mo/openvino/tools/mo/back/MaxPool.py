# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from openvino.tools.mo.back.FakeOutputResolver import FakeOutputResolver
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.result import Result


class MaxPool(BackReplacementPattern):
    """
    Rename Pooling/max to MaxPool
    """
    enabled = True

    def run_after(self):
        return [FakeOutputResolver]

    def pattern(self):
        return dict(
            nodes=[
                ('pooling', {'type': 'Pooling', 'pool_method': 'max'})
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['pooling']
        node.type = 'MaxPool'
        del node['pool_method']
        if 'exclude_pad' in node:
            del node['exclude_pad']

        # adding missed outputs for MaxPool node
        MaxPool.normalize_outputs(node)

    @staticmethod
    def normalize_outputs(node: Node):
        if node.out_port(0).disconnected():
            output = Result(node.graph, {'name': node.name + '/Result_port_0/',
                                         'keep_output_port': node.has_and_set('remove_values_output')}).create_node()
            node.out_port(0).get_connection().set_destination(output.in_port(0))

        # we check port existing to support MaxPool_1 with only 1 output port and MaxPool_8 with 2 output ports
        if node.has_port('out', 1) and node.out_port(1).disconnected():
            output = Result(node.graph, {'name': node.name + '/Result_port_1/',
                                         'keep_output_port': node.has_and_set('remove_values_output')}).create_node()
            node.out_port(1).get_connection().set_destination(output.in_port(0))
