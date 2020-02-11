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

from mo.utils.ir_reader.extender import Extender
from mo.graph.graph import Node

from mo.front.common.partial_infer.utils import int64_array


class AvgPool_extender(Extender):
    op = 'AvgPool'

    @staticmethod
    def extend(op: Node):
        common_pool_extender(op)

        if 'exclude-pad' in op:
            op['exclude_pad'] = op['exclude-pad']
            del op['exclude-pad']


class MaxPool_extender(Extender):
    op = 'MaxPool'

    @staticmethod
    def extend(op: Node):
        common_pool_extender(op)


def common_pool_extender(op: Node):
    op['stride'] = int64_array([1, 1] + op.strides)
    op['window'] = int64_array([1, 1] + op.kernel)
    op['kernel_spatial'] = op.kernel
    op['output_spatial_shape'] = None

    op['batch_dims'] = int64_array([0]),
    op['channel_dims'] = int64_array([1]),

    dim = len(op.pads_begin)

    assert dim in (2, 3), '{}D {} not supported!'.format(dim, op.op)

    pad = [[0, 0], [0, 0]]
    pad.extend([[op.pads_begin[i], op.pads_end[i]] for i in range(dim)])

    op['pad'] = int64_array(pad)

    op['spatial_dims'] = [i + 2 for i in range(dim)]

    if op.has_valid('rounding_type') and op.rounding_type == 'ceil':
        op['pooling_convention'] = 'full'
