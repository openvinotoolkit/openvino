# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.tf.mvn_unrolled import MVNUnrolled
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph_with_attrs


class MVNUnrolledMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['MVN'] = MVN

    def test(self):
        pattern_matcher = MVNUnrolled()
        pattern = pattern_matcher.pattern()
        graph = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'], edges_with_attrs=pattern['edges'], update_edge_attrs=None,
                                       new_nodes_with_attrs=[('reduction_indicies', {'kind': 'data'}),
                                                             ('conv2d', {'kind': 'op'}),
                                                             ('variance_reduction', {'kind': 'data'}),
                                                             ('pow2', {'kind': 'data'}),
                                                             ('eps', {'kind': 'data'}),
                                                             ('next_op', {'kind': 'op'})],
                                       new_edges_with_attrs=[('reduction_indicies', 'mean', {'in': 1}),
                                                             ('conv2d', 'mean',{'in': 0, 'out': 1}),
                                                             ('variance_reduction', 'variance', {'in': 1}),
                                                             ('pow2', 'pow', {'in': 1}),
                                                             ('eps', 'add'), ('truediv', 'next_op')])
        graph.graph['layout'] = 'NHWC'
        pattern_matcher.find_and_replace_pattern(graph)

        graph_ref = build_graph_with_attrs(nodes_with_attrs=pattern['nodes'][:-1],
                                           edges_with_attrs=pattern['edges'][:-2], update_edge_attrs=None,
                                           new_nodes_with_attrs=[('reduction_indicies', {'kind':'data'}),
                                                                 ('conv2d', {'kind':'op'}),
                                                                 ('variance_reduction', {'kind':'data'}),
                                                                 ('pow2', {'kind': 'data'}),
                                                                 ('eps', {'kind': 'data'}),
                                                                 ('mvn', {'kind': 'op', 'op': 'MVN'}),
                                                                 ('next_op', {'kind': 'op'})],
                                           new_edges_with_attrs=[('reduction_indicies', 'mean', {'in':1}),
                                                                 ('conv2d', 'mean', {'in': 0}),
                                                                 ('variance_reduction', 'variance',{'in': 1}),
                                                                 ('pow2', 'pow', {'in': 1}),
                                                                 ('eps', 'add'),
                                                                 ('conv2d', 'mvn',{'in': 0}),
                                                                 ('reduction_indicies', 'mvn', {'in': 1}),
                                                                 ('variance_reduction', 'mvn',{'in': 2}),
                                                                 ('pow2', 'mvn', {'in': 3}),
                                                                 ('eps', 'mvn',{'in': 4}),
                                                                 ('mvn', 'next_op')])

        (flag, resp) = compare_graphs(graph, graph_ref, 'next_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
