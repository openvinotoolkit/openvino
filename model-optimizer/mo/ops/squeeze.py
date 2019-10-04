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

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op
from mo.utils.error import Error


class Squeeze(Op):
    op = 'Squeeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'op': __class__.op,
            'type': __class__.op,
            'squeeze_dims': None,
            'reinterp_shape': True,
            'keep_at_least_1d': 0,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        real_squeeze_dims = int64_array([])
        input_shape = node.in_node().shape
        if input_shape is None:
            return

        output_shape = input_shape.copy()
        assert len(node.in_nodes()) == 2, 'The Squeeze node {} must have 2 inputs'.format(node.soft_get('name'))

        # TODO remove the following 'if' statement when IE start support 0D tensors
        squeeze_dims = node.in_port(1).data.get_value()
        if squeeze_dims.ndim == 0:
            squeeze_dims = squeeze_dims.reshape([1])

        for dim in squeeze_dims:
            if output_shape[dim] == 1:
                real_squeeze_dims = np.append(real_squeeze_dims, get_canonical_axis_index(output_shape, dim))
            else:
                raise Error('Trying to squeeze dimension not equal to 1 for node "{}"'.format(node.soft_get('name')))

        output_shape = np.delete(output_shape, real_squeeze_dims)
        node.out_node().shape = output_shape

        # make dimensions positive to correctly translate from NHWC to NCHW layout
        if node.in_port(1).get_source().node.op == 'Const':
            node.in_port(1).data.set_value(real_squeeze_dims)

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value().reshape(output_shape))

        # the squeeze_dim attribute will be converted to the second input in the end of the Middle phase
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')
