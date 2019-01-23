"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from extensions.ops.instance_normalization import InstanceNormalization


class InstanceNormalizationOp(unittest.TestCase):
    def test_constructor_supported_attrs(self):
        graph = nx.MultiDiGraph()
        op = InstanceNormalization(graph, attrs={'epsilon': 0.1})
        self.assertEqual(op.supported_attrs(), ['epsilon'])
