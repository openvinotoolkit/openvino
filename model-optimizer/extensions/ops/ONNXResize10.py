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

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ONNXResize10Op(Op):
    op = 'ONNXResize10'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': ONNXResize10Op.onnx_resize_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return ['mode']

    @staticmethod
    def onnx_resize_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        scale_value = node.in_port(1).data.get_value()
        assert scale_value is not None, \
            "Node {} with op {} has no scales".format(node.soft_get('name', node.id), node.op)
        scale = np.array(scale_value)
        output_shape = np.floor(input_shape * scale + 1.0e-6).astype(np.int64)

        node.out_port(0).data.set_shape(output_shape.copy())
