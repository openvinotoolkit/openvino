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
import unittest

from extensions.front.tf.mvn_unrolled import MVNUnrolled
from mo.ops.op import Op
from mo.utils.unittest.graph import compare_graphs, build_graph_with_attrs
from extensions.ops.mvn import MVN


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
