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
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class RemoveIdentity(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', identity=True))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])


class RemoveDropout(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(op='Dropout'))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])


class RemoveNodesWithZeroPhase(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', phase=0))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        remove_op_node_with_data_node(graph, match['op'])
