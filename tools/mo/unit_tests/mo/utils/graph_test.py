# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.graph import bfs_search, is_connected_component, sub_graph_between_nodes, backward_bfs_for_operation
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import regular_op, result, build_graph_with_edge_attrs


class TestGraphUtils(UnitTestWithMockedTelemetry):
    def test_simple_dfs(self):
        graph = Graph()
        graph.add_nodes_from(list(range(1, 5)))
        graph.add_edges_from([(1, 2), (1, 3), (3, 4)])

        visited = set()
        order = graph.dfs(1, visited)
        self.assertTrue(order == [4, 3, 2, 1] or order == [2, 4, 3, 1])

    def test_bfs_search_default_start_nodes(self):
        """
        Check that BFS automatically determines input nodes and start searching from them.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5)])

        order = bfs_search(graph)
        self.assertTrue(order == [1, 2, 3, 4, 5] or order == [2, 1, 3, 4, 5])

    def test_bfs_search_specific_start_nodes(self):
        """
        Check that BFS stars from the user defined nodes and doesn't go in backward edge direction.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5), (6, 1)])

        order = bfs_search(graph, [1])
        self.assertTrue(order == [1, 3, 4, 5])

    def test_is_connected_component_two_separate_sub_graphs(self):
        """
        Check that if there are two separate sub-graphs the function returns False.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])
        self.assertFalse(is_connected_component(graph, list(range(1, 7))))
        self.assertFalse(is_connected_component(graph, [1, 3]))
        self.assertFalse(is_connected_component(graph, [6, 4]))
        self.assertFalse(is_connected_component(graph, [2, 5]))

    def test_is_connected_component_two_separate_sub_graphs_divided_by_ignored_node(self):
        """
        Check that if there are two separate sub-graphs the function connected by an edge going through the ignored node
        then the function returns False.
        """
        graph = Graph()
        node_names = list(range(1, 8))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (7, 4)])
        self.assertFalse(is_connected_component(graph, list(range(1, 7))))

    def test_is_connected_component_connected(self):
        """
        Check that if the sub-graph is connected.
        """
        graph = Graph()
        node_names = list(range(1, 8))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6), (1, 7), (7, 4)])
        self.assertTrue(is_connected_component(graph, list(range(1, 8))))

    def test_is_connected_component_edges_direction_is_ignored(self):
        """
        Check that edges direction is ignored when checking for the connectivity.
        """
        graph = Graph()
        node_names = list(range(1, 5))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(2, 1), (2, 3), (4, 3)])
        self.assertTrue(is_connected_component(graph, node_names))
        self.assertTrue(is_connected_component(graph, [2, 1]))
        self.assertTrue(is_connected_component(graph, [4, 2, 3]))

    def test_is_connected_component_edges_direction_is_ignored_not_connected(self):
        """
        Check that edges direction is ignored when checking for the connectivity. In this case the graph is not
        connected.
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 5)))
        graph.add_edges_from([(2, 1), (2, 3), (4, 3)])
        self.assertFalse(is_connected_component(graph, [1, 2, 4]))
        self.assertFalse(is_connected_component(graph, [1, 4]))
        self.assertFalse(is_connected_component(graph, [2, 4]))
        self.assertFalse(is_connected_component(graph, [3, 4, 1]))

    def test_sub_graph_between_nodes_include_incoming_edges_for_internal_nodes(self):
        """
        Check that the function adds input nodes for the internal nodes of the graph. For example, we need to add node 5
        and 6 in the case below if we find match from node 1 till node 4.
        6 -> 5 ->
                 \
            1 -> 2 -> 3 -> 4
        :return:
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), list(range(1, 7)))

        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [2])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [1, 2, 5, 6])

    def test_sub_graph_between_nodes_do_not_include_incoming_edges_for_input_nodes(self):
        """
        Check that the function doesn't add input nodes for the start nodes of the sub-graph. For example, we do not
        need to add node 5 in the case below if we find match from node 1 till node 4.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [2, 3, 4])

    def test_sub_graph_between_nodes_placeholder_included(self):
        """
        Check that the function doesn't allow to add Placeholders to the sub-graph. 5 is the Placeholder op.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.node[5]['op'] = 'Parameter'
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        self.assertRaises(Error, sub_graph_between_nodes, graph, [1], [4])

    def test_sub_graph_between_nodes_placeholder_excluded(self):
        """
        Check that the function do not check that node is Placeholders for the nodes not included into the sub-graph.
        For example, node 5 is Placeholder but it is not included into the sub-graph, so this attribute is ignored.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.node[5]['op'] = 'Parameter'
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), [2, 3, 4])

    def test_sub_graph_between_nodes_multiple_inputs(self):
        """
        Check that the function works correctly when multiple inputs specified.
          5->
             \
        1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 6)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [2, 5], [4])
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([2, 3, 4, 5]))

    def test_sub_graph_between_nodes_branches_included(self):
        """
        Check that the function works correctly for tree like structures.
        1 -> 2 -> 3 -> 4
             \
             5 -> 6
            / \
        9 ->   -> 7 -> 8
        """
        graph = Graph()
        node_names = list(range(1, 10))
        graph.add_nodes_from(node_names)
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (5, 7), (7, 8), (9, 5)])
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [4])), node_names)
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [6])), node_names)
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [8])), node_names)
        # all nodes except 4 because it is a child of end node
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [1], [3])), [n for n in node_names if n != 4])
        # all nodes except 1 because it is a parent node child of start node. The nodes 3 and 4 must be added because
        # after merging node 2 into sub-graph the node 2 will be removed and it is not known how to calculate the tensor
        # between node 2 and 3.
        self.assertListEqual(sorted(sub_graph_between_nodes(graph, [2], [8])), [n for n in node_names if n != 1])

    def test_sub_graph_between_nodes_control_flow_included(self):
        """
        Check that the function works correctly for case when control flow edges must be traversed (edge 5 -> 2).
        6 -> 5->
                \
           1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2, {'control_flow_edge': True}), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=True)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4, 5, 6]))

    def test_sub_graph_between_nodes_control_flow_not_included(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 5 -> 2).
        6 -> 5->
                \
           1 -> 2 -> 3 -> 4
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (5, 2, {'control_flow_edge': True}), (6, 5)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=False)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4]))

    def test_sub_graph_between_nodes_control_flow_included_forward(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 3 -> 5).
           1 -> 2 -> 3 -> 4
                      \
                       -> 5 -> 6
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5, {'control_flow_edge': True}), (5, 6)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=True)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4, 5, 6]))

    def test_sub_graph_between_nodes_control_flow_not_included_forward(self):
        """
        Check that the function works correctly for case when control flow edges should not be traversed (edge 3 -> 5).
           1 -> 2 -> 3 -> 4
                      \
                       -> 5 -> 6
        """
        graph = Graph()
        graph.add_nodes_from(list(range(1, 7)))
        graph.add_edges_from([(1, 2), (2, 3), (3, 4), (3, 5, {'control_flow_edge': True}), (5, 6)])
        sub_graph_nodes = sub_graph_between_nodes(graph, [1], [4], include_control_flow=False)
        self.assertIsNotNone(sub_graph_nodes)
        self.assertListEqual(sorted(sub_graph_nodes), sorted([1, 2, 3, 4]))

    def test_backward_bfs_for_op_no_ops_detected(self):
        nodes = {**regular_op('input', {'op': 'Parameter'}),
                 **regular_op('hsigmoid', {'op': 'HSigmoid'}),
                 **result('result'),
                 }
        edges = [('input', 'hsigmoid', {'out': 0, 'in': 0}),
                 ('hsigmoid', 'result', {'out': 0, 'in': 0}),
                 ]

        graph = build_graph_with_edge_attrs(nodes, edges)
        graph.stage = 'front'

        found_nodes = backward_bfs_for_operation(Node(graph, 'result'), ['NonExistingOp'])
        self.assertEqual(len(found_nodes), 0)

    def test_backward_bfs_for_op_closest_op_detected(self):
        """
        input -> hsigmoid_1 -> hsigmoid_2 -> result
        The returned op should be first met HSigmoid which is hsigmoid_2
        """
        nodes = {**regular_op('input', {'op': 'Parameter'}),
                 **regular_op('hsigmoid_1', {'op': 'HSigmoid'}),
                 **regular_op('hsigmoid_2', {'op': 'HSigmoid'}),
                 **result('result'),
                 }
        edges = [('input', 'hsigmoid_1', {'out': 0, 'in': 0}),
                 ('hsigmoid_1', 'hsigmoid_2', {'out': 0, 'in': 0}),
                 ('hsigmoid_2', 'result', {'out': 0, 'in': 0}),
                 ]

        graph = build_graph_with_edge_attrs(nodes, edges)
        graph.stage = 'front'

        found_nodes = backward_bfs_for_operation(Node(graph, 'result'), ['HSigmoid'])
        self.assertEqual(len(found_nodes), 1)
        self.assertEqual(found_nodes[0].id, 'hsigmoid_2')

    def test_backward_bfs_for_op_parallel_branch_op_detected(self):
        r"""
        input_1 -> hsigmoid_1 -> hsigmoid_2 ->
                                               \
                                                - Concat->result
                                               /
        input_2 -> hsigmoid_3 -> hsigmoid_4 ->
        The returned op should be first met HSigmoids which are hsigmoid_2 and hsigmoid_4
        """
        nodes = {**regular_op('input_1', {'op': 'Parameter'}),
                 **regular_op('hsigmoid_1', {'op': 'HSigmoid'}),
                 **regular_op('hsigmoid_2', {'op': 'HSigmoid'}),
                 **regular_op('input_2', {'op': 'Parameter'}),
                 **regular_op('hsigmoid_3', {'op': 'HSigmoid'}),
                 **regular_op('hsigmoid_4', {'op': 'HSigmoid'}),
                 **regular_op('concat', {'op': 'Concat'}),
                 **result('result'),
                 }
        edges = [('input_1', 'hsigmoid_1', {'out': 0, 'in': 0}),
                 ('hsigmoid_1', 'hsigmoid_2', {'out': 0, 'in': 0}),
                 ('hsigmoid_2', 'concat', {'out': 0, 'in': 0}),
                 ('input_2', 'hsigmoid_3', {'out': 0, 'in': 0}),
                 ('hsigmoid_3', 'hsigmoid_4', {'out': 0, 'in': 0}),
                 ('hsigmoid_4', 'concat', {'out': 0, 'in': 1}),
                 ('concat', 'result', {'out': 0, 'in': 0}),
                 ]

        graph = build_graph_with_edge_attrs(nodes, edges)
        graph.stage = 'front'

        found_nodes = backward_bfs_for_operation(Node(graph, 'result'), ['HSigmoid'])
        self.assertEqual(len(found_nodes), 2)
        self.assertSetEqual({found_nodes[0].id, found_nodes[1].id}, {'hsigmoid_2', 'hsigmoid_4'})

    def test_backward_bfs_for_op_parallel_branch_stop_op(self):
        r"""
        input_1 -> hsigmoid_1 -> hsigmoid_2 ->
                                               \
                                                - Concat->result
                                               /
        input_2 -> hsigmoid_3 -> ShapeOf    ->
        The returned op should be first met HSigmoids which is hsigmoid_2, but not the hsigmoid_3 located after banned
        operation of type "ShapeOf"
        """
        nodes = {**regular_op('input_1', {'op': 'Parameter'}),
                 **regular_op('hsigmoid_1', {'op': 'HSigmoid'}),
                 **regular_op('hsigmoid_2', {'op': 'HSigmoid'}),
                 **regular_op('input_2', {'op': 'Parameter'}),
                 **regular_op('hsigmoid_3', {'op': 'HSigmoid'}),
                 **regular_op('shapeof', {'op': 'ShapeOf'}),
                 **regular_op('concat', {'op': 'Concat'}),
                 **result('result'),
                 }
        edges = [('input_1', 'hsigmoid_1', {'out': 0, 'in': 0}),
                 ('hsigmoid_1', 'hsigmoid_2', {'out': 0, 'in': 0}),
                 ('hsigmoid_2', 'concat', {'out': 0, 'in': 0}),
                 ('input_2', 'hsigmoid_3', {'out': 0, 'in': 0}),
                 ('hsigmoid_3', 'shapeof', {'out': 0, 'in': 0}),
                 ('shapeof', 'concat', {'out': 0, 'in': 1}),
                 ('concat', 'result', {'out': 0, 'in': 0}),
                 ]

        graph = build_graph_with_edge_attrs(nodes, edges)
        graph.stage = 'front'

        found_nodes = backward_bfs_for_operation(Node(graph, 'result'), ['HSigmoid'], ['ShapeOf'])
        self.assertEqual(len(found_nodes), 1)
        self.assertEqual(found_nodes[0].id, 'hsigmoid_2')
