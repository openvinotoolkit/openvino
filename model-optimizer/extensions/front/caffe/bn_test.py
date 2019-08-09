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
import numpy as np
import unittest

from extensions.front.caffe.bn import BNToScaleShift
from mo.graph.graph import Node
from mo.utils.unittest.extractors import FakeParam
from mo.utils.unittest.graph import build_graph_with_edge_attrs


class FakeBNProtoLayer:
    def __init__(self, val):
        self.bn_param = val


class FakeBNBinLayer:
    def __init__(self, val):
        self.blobs = val


class TestBNReplacer(unittest.TestCase):
    def test_bn(self):
        bn_pb = FakeBNProtoLayer(FakeParam('eps', 0.0001))
        mean = [1, 2.5, 3]
        var = [0.5, 0.1, 1.2]
        scale = [2.3, 3.4, 4.5]
        shift = [0.8, 0.6, 0.4]
        bn_bin = FakeBNBinLayer([FakeParam('data', mean),
                                 FakeParam('data', var),
                                 FakeParam('data', scale),
                                 FakeParam('data', shift)])
        nodes = {
            'node_1': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'},
            'bn': {'type': 'BN', 'kind': 'op', 'op': 'BN',
                   'pb': bn_pb,
                   'model_pb': bn_bin},
            'node_2': {'kind': 'op', 'type': 'Identity', 'op': 'Parameter'}}
        edges = [
            ('node_1', 'bn', {'in': 0}),
            ('bn', 'node_2', {'in': 0})]
        graph = build_graph_with_edge_attrs(nodes, edges)
        node = Node(graph, 'bn')
        replacer = BNToScaleShift()
        replacer.replace_op(graph, node)

        scale_node = [node for node, attrs in list(graph.nodes(data=True)) if attrs['type'] == 'ScaleShift']
        self.assertEqual(len(scale_node), 1)

        scale_ref = np.array([1.11796412, 3.2272172, 4.74282367])
        shift_ref = np.array([-2.07131747, -10.87253847, -20.14270653])
        for i in range(len(mean)):
            self.assertAlmostEqual(graph.node[scale_node[0]]['scale'][i], scale_ref[i])
            self.assertAlmostEqual(graph.node[scale_node[0]]['bias'][i], shift_ref[i])
