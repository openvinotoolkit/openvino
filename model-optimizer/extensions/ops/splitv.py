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

import networkx as nx

from mo.front.common.partial_infer.split import tf_split_v_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class SplitV(Op):
    op = 'SplitV'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': 'Split',
            'op': 'SplitV',
            'axis': 1,
            'input_port': 0,
            'in_ports_count': 3,
            'infer': tf_split_v_infer
        }, attrs)

    def supported_attrs(self):
        return ['axis', 'size_splits']

    def backend_attrs(self):
        return ['axis']
