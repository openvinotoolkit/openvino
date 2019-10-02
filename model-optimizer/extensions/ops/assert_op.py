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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error


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
        node.out_node().value = assert_value
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

