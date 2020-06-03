"""
 Copyright (C) 2020 Intel Corporation

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
import logging as log

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class RemoveConstToResult(BackReplacementPattern):
    """
    Transformation looks for a sub-graph "Const->Result"
    and removes Result node.
    """
    enabled = True
    #force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('const_node', {'type': 'Const', 'kind': 'op'}),
                ('const_data', {'kind': 'data'}),
                ('result_node', {'type': 'Result', 'kind': 'op'}),
            ],
            edges=[
                ('const_node', 'const_data'),
                ('const_data', 'result_node')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        result_node = match['result_node']
        graph.remove_node(result_node.id)
