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

from mo.front.common.partial_infer.utils import int64_array
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class ExtractImagePatches(Extender):
    op = 'ExtractImagePatches'

    @staticmethod
    def extend(op: Node):
        op['sizes'] = int64_array([1, 1] + op.sizes)
        op['strides'] = int64_array([1, 1] + op.strides)
        op['rates'] = int64_array([1, 1] + op.rates)

        op['batch_dims'] = int64_array([0])
        op['channel_dims'] = int64_array([1])
        op['spatial_dims'] = int64_array([2, 3])
