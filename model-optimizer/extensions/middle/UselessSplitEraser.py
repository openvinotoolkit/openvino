"""
 Copyright (C) 2018-2020 Intel Corporation

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


class UselessSplitEraser(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('split', {'op': 'Split', 'num_splits': 1})],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['split']
        name = node.soft_get('name', node.id)

        assert node.soft_get('input_port', 0) == 0, \
            'Internal attribute `input_port` was not resolved on front phase, broken Split {}'.format(name)
        assert len(node.out_ports()) == 1

        node.out_port(0).get_connection().set_source(node.in_port(0).get_connection().get_source())
