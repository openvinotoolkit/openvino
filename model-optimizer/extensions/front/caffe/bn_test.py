"""
 Copyright (C) 2018-2021 Intel Corporation

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

import numpy as np

from extensions.front.caffe.bn import BNToScaleShift
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.extractors import FakeParam
from mo.utils.unittest.graph import build_graph_with_edge_attrs, build_graph_with_attrs


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
        nodes = [
            ('input', {'kind': 'op', 'type': 'Identity', 'op': 'Identity'}),
            ('bn', {'type': None, 'kind': 'op', 'op': 'BatchNormInference', 'pb': bn_pb, 'model_pb': bn_bin}),
            ('output', {'kind': 'op', 'type': 'Identity', 'op': 'Identity'}),
        ]
        edges = [
            ('input', 'bn', {'in': 0, 'out': 0}),
            ('bn', 'output', {'in': 0, 'out': 0}),
        ]
        graph = build_graph_with_attrs(nodes, edges)
        graph.stage = 'front'

        BNToScaleShift().find_and_replace_pattern(graph)

        ref_nodes = {
            'input': {'kind': 'op', 'type': 'Identity', 'op': 'Identity'},
            'scale': {'kind': 'op', 'type': 'Const', 'op': 'Const',
                      'value': np.array([1.11796412, 3.2272172, 4.74282367])},
            'shift': {'kind': 'op', 'type': 'Const', 'op': 'Const',
                      'value': np.array([-2.07131747, -10.87253847, -20.14270653])},
            'ss': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
            'output': {'kind': 'op', 'type': 'Identity', 'op': 'Identity'},
        }
        ref_edges = [
            ('input', 'ss', {'in': 0, 'out': 0}),
            ('scale', 'ss', {'in': 1, 'out': 0}),
            ('shift', 'ss', {'in': 2, 'out': 0}),
            ('ss', 'output', {'in': 0, 'out': 0}),
        ]
        ref_graph = build_graph_with_edge_attrs(ref_nodes, ref_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'input', check_op_attrs=True)
        self.assertTrue(flag, resp)
