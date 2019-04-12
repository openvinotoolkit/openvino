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

from extensions.front.kaldi.fuse_repeated_reshape import FuseRepeatedReshapes
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node


class EliminateRedundantReshape(FrontReplacementPattern):
    enabled = False

    def run_after(self):
        return [
            FuseRepeatedReshapes
        ]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('reshape', dict(kind='op', op='Reshape'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape_node = match['reshape']
        in_node = reshape_node.in_node()
        out_node = reshape_node.out_node()
        if not np.array_equal(in_node.shape, out_node.shape):
            return False
        remove_op_node_with_data_node(graph, reshape_node)
