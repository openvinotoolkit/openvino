# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.Exit import Exit
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, shaped_parameter


# test for TensorIterator
graph_nodes = {
    **shaped_parameter("input", int64_array([1, 4, 64, 54])),
    **regular_op_with_empty_data("exit", {'op': "Exit"}),
    **result("output")
}


class ExitTest(unittest.TestCase):
    def test_exit_static(self):
        graph = build_graph(nodes_attrs=graph_nodes,
                            edges=[*connect('input', 'exit'),
                                   *connect('exit', 'output')],
                            nodes_with_edges_only=True)
        exit_node = Node(graph, 'exit')
        in_node = Node(graph, 'input')

        Exit.exit_infer(exit_node)

        self.assertTrue(np.ma.allequal(exit_node.out_port(0).data.get_shape(), in_node.shape))

    def test_exit_dynamic(self):
        graph = build_graph(nodes_attrs=graph_nodes,
                            edges=[*connect('input', 'exit'),
                                   *connect('exit', 'output')],
                            nodes_with_edges_only=True)
        exit_node = Node(graph, 'exit')
        in_node = Node(graph, 'input')
        shape = int64_array([-1, 36])
        in_node.shape = np.ma.masked_array(shape, mask=shape == -1, fill_value=dynamic_dimension_value)
        in_node.out_port(0).data.set_shape(in_node.shape)

        Exit.exit_infer(exit_node)

        self.assertTrue(np.ma.allequal(exit_node.out_port(0).data.get_shape(), in_node.shape))
