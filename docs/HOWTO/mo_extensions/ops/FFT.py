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

#! [fft:operation]
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class FFT(Op):
    op = 'FFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'custom_opset',
            'inverse': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': copy_shape_infer
        }, attrs)

    def backend_attrs(self):
        return ['inverse']
#! [fft:operation]
