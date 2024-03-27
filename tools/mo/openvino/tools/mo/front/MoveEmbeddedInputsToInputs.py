# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.pass_separator import FrontStart
from openvino.tools.mo.front.restore_ports import RestorePorts
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class MoveEmbeddedInputsToInputs(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        return [RestorePorts]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(kind='op', embedded_inputs=lambda x: x is not None))],
            edges=[]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        node = match['op']
        for port_index, value_attr, attrs in node['embedded_inputs']:
            const = Const(graph, dict(value=node[value_attr])).create_node()
            node.add_input_port(port_index, skip_if_exist=True)
            const.out_port(0).connect(node.in_port(port_index))
            node.in_port(port_index).bin = attrs['bin']
            node.in_port(port_index).in_attrs.append('bin')
            del node[value_attr]
        del node['embedded_inputs']
