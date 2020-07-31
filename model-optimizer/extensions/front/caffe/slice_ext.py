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
from mo.front.extractor import FrontExtractorOp
from mo.ops.slice import CaffeSlice


class SliceFrontExtractor(FrontExtractorOp):
    op = 'slice'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.slice_param

        # slice_dim is deprecated parameter and is used as alias for axis
        # however if slice_dim is defined and axis is default, we use slice_dim
        if param.slice_dim != 1 and param.axis == 1:
            axis = param.slice_dim
        else:
            axis = param.axis

        update_attrs = {
            'axis': axis,
            'slice_point': int64_array(param.slice_point),
            'in_ports_count': 1,
            'out_ports_count': len(param.slice_point) + 1,
        }

        CaffeSlice.update_node_stat(node, update_attrs)
        return cls.enabled
