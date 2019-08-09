"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class WeightsPermuteNormalizer(MiddleReplacementPattern):
    """
    We propagate PermuteAttr from weights port of Convolution and FullyConnected to real Const that contains it
    """
    enabled = True

    def run_after(self):
        from extensions.middle.GemmToFullyConnected import GemmToFullyConnected
        return [GemmToFullyConnected]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('const_data', dict(kind='data')),
                ('const', dict(type='Const')),
                ('quantize', dict(type='FakeQuantize')),
                ('quantize_data', dict(kind='data')),
                ('conv', dict(type=lambda type: type in ['Convolution', 'FullyConnected', 'MatMul'])),
            ],
            edges=[
                ('const', 'const_data'),
                ('const_data', 'quantize', {'in': 0}),
                ('quantize', 'quantize_data'),
                ('quantize_data', 'conv', {'in': 1}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        conv = match['conv']
        if 1 not in conv.in_edges() or 'permutation' not in conv.in_edge(1):
            return

        perm = conv.in_edge(1)['permutation']
        match['quantize'].in_port(0).permutation = perm
