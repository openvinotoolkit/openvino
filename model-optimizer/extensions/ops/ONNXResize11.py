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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ONNXResize11Op(Op):
    op = 'ONNXResize11'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'version': 'opset3',
            'out_ports_count': 1,
            'infer': ONNXResize11Op.onnx_resize_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'coordinate_transformation_mode',
            'cubic_coeff_a',
            'exclude_outside',
            'extrapolation_value',
            'mode',
            'nearest_mode'
        ]

    @staticmethod
    def onnx_resize_infer(node: Node):

        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        num_of_in_nodes = len(node.in_nodes())
        assert num_of_in_nodes in {3, 4}, \
            "Node {} with op {} number of inputs must be equal to 3 or 4.".format(node.name, node.op)

        assert node.coordinate_transformation_mode != 'tf_crop_and_resize', \
            'Mode tf_crop_and_resize is not supported for op {} with name {}'.format(node.op, node.name)

        if num_of_in_nodes == 3:
            # i.e. input 'sizes' is not given
            assert node.in_node(2).has_valid('value'), \
                "Node {} with op {} has no value in input port 2".format(node.name, node.op)
            scale = int64_array(node.in_node(2).value)
            output_shape = np.floor(input_shape * scale).astype(np.int64)
        else:
            # i.e. input 'sizes' is given
            assert node.in_node(3).has_valid('value'), \
                "Node {} with op {} has no value in input port 3".format(node.name, node.op)
            output_shape = node.in_node(3).value

        node.out_node().shape = output_shape
