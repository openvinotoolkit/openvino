# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Assert(Op):
    op = 'Assert'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'infer': Assert.assert_infer,
            'cf_infer': Assert.assert_control_flow_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def assert_infer(node: Node):
        assert_value = node.in_node(0).value
        node.out_node().value = assert_value.copy()
        node.out_node().shape = []

    @staticmethod
    def assert_control_flow_infer(node: Node,  is_executable: bool, mark_executability: callable):
        """
        Infers control flow through assert operation node. It marks output data nodes executability according to
        executability of current node and assert data value
        :param node: Node instance to infer control flow through
        :param is_executable: if current node is executable
        :param mark_executability: function to mark executability of node
        """
        graph = node.graph
        assert_value = node.out_node().value
        for n in [v for _, v in graph.out_edges(node.id)]:
            mark_executability(n, assert_value and is_executable)

