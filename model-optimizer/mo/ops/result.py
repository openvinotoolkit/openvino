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

from mo.graph.graph import Graph
from mo.ops.op import Op


class Result(Op):
    """
    Operation that should be added after the output node of the graph. It is a marker of the graph output.
    This type of nodes is used in the dead nodes elimination pass and not dumped into the IR.
    """
    op = 'Result'

    def __init__(self, graph: Graph, attrs: dict = None):
        super().__init__(graph, {
            'op': __class__.op,
            'type': __class__.op,
            'infer': lambda x: None,
            'value': None,
            'data_type': None,
            'in_ports_count': 1,
        }, attrs)
