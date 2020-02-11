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
from mo.utils.graph import Node

from mo.front.common.partial_infer.utils import int64_array


class Conv_extender(Extender):
    op = 'Convolution'

    @staticmethod
    def extend(op: Node):
        for attr in ['strides', 'dilations', 'pads_begin', 'pads_end', 'output_padding']:
            Extender.attr_to_list(op, attr)

        op['stride'] = int64_array([1, 1] + op.strides)
        op['dilation'] = int64_array([1, 1] + op.dilations)

        op['batch_dims'] = int64_array([0])
        op['channel_dims'] = int64_array([1])

        # Be VERY careful with these attributes!
        op['input_feature_channel'] = 1
        op['output_feature_channel'] = 0

        dim = len(op.pads_begin)

        assert dim in (1, 2, 3), '{}D Convolution not supported!'.format(dim)

        pad = [[0, 0], [0, 0]]
        pad.extend([[op.pads_begin[i], op.pads_end[i]] for i in range(dim)])

        op['pad'] = int64_array(pad)

        op['spatial_dims'] = [i + 2 for i in range(dim)]
