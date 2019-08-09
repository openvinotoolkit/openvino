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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class LRNToNorm(BackReplacementPattern):
    """
    For IR versions lesser than IR v10, transform LRN back to Norm
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('lrn', dict(kind='op', op='LRN'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['lrn']
        if graph.graph['cmd_params'].generate_experimental_IR_V10:
            if node.has_valid('region'):
                assert node.region == 'across', 'LRN region != across is temporary unsupported case'
                del node['region']
        else:
            if not node.has_valid('region'):
                node['region'] = 'across'
            if node.has_valid('bias'):
                # node.bias != 1 should be eliminated before, if not, it is an internal bug
                assert node.bias == 1
                del node['bias']
            node['type'] = 'Norm'
