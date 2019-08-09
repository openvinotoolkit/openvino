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
from mo.ops.convolution import Convolution
from mo.ops.op import Op


class DeformableConvolution(Op):
    op = 'DeformableConvolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'infer': Convolution.infer,
            'multiplication_transparent': True,
            'multiplication_transparent_ports': [(0, 0), (1, 0)],
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        # the same attributes as in a regular convolution and one additional attribute 'deformable_group'
        return Convolution(self.graph, {}).backend_attrs() + ['deformable_group']
