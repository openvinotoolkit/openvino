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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class LRN(Op):
    op = 'LRN'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': 'LRN',
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'bias': 1,
            'infer': copy_shape_infer
        }, attrs)

    def supported_attrs(self):
        return [
            'bias',    # supported in V10 only; should be eliminated for other versions
            'alpha',
            'beta',
            ('local-size', lambda node: node.local_size),
            'region'   # deprecated in V10 attribute, but it is kept for V6 compatibility
        ]
