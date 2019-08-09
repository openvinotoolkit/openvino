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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.middle.passes.fusing.helpers import get_tensor_in_port, get_value_in_port
from mo.middle.replacement import MiddleReplacementPattern


class EltwiseChecker(MiddleReplacementPattern):
    """
    Checks if element-wise operation can be converted to ScaleShift or not:
        decision gets made by verifying constant input value shape is like 1,N,1,1
    """
    enabled = True

    def run_after(self):
        from extensions.middle.EltwiseInputReshape import Eltwise1DInputReshape
        from extensions.middle.GemmToFullyConnected import GemmToFullyConnected
        return [Eltwise1DInputReshape, GemmToFullyConnected]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @staticmethod
    def set_flags_to_false(node: Node, flags: list):
        for flag in flags:
            node[flag] = False

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(is_eltwise=True):

            tensor_port, value_port = get_tensor_in_port(node), get_value_in_port(node)
            if tensor_port is None or value_port is None:
                self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
                continue

            tensor_shape = tensor_port.data.get_shape()
            out_shape = node.out_port(0).data.get_shape()
            assert tensor_shape is not None and out_shape is not None
            if not np.array_equal(tensor_shape, out_shape):
                # ScaleShift operation doesn't support broadcasting
                self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
                continue

            value_shape = value_port.data.get_shape()
            assert value_shape is not None
            assert len(value_shape) <= len(tensor_shape), \
                "No broadcasting was done for elementwise node {} due to previous checks in EltwiseChecker class. " \
                "But constant input rank is larger than tensor input rank, that is inconsistent".format(node.name)

            broadcasted_value_shape = np.insert(value_shape, 0, [1] * (len(tensor_shape) - len(value_shape)))

            feature_dim = min(1, tensor_shape.size - 1) if node.graph.graph['layout'] == 'NCHW' else -1
            ones = np.ones(len(tensor_shape))
            possible_shape = ones.copy()
            np.put(possible_shape, feature_dim, tensor_shape.item(feature_dim))

            if not np.array_equal(broadcasted_value_shape, ones) and \
                    not np.array_equal(broadcasted_value_shape, possible_shape):
                # ScaleShift weights should have [1,C,1,1]-like or [1,1,1,1]-like shape
                self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
                continue

            if len(tensor_shape) not in [2, 4, 5]:
                # ScaleShift operation is supported for 2D, 4D or 5D tensor inputs
                self.set_flags_to_false(node, ['can_be_scaleshift'])
                continue
