"""
 Copyright (c) 2018 Intel Corporation

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

from mo.ops.op import Op
from mo.front.common.partial_infer.expand_dims import tf_expand_dims_infer


class ExpandDims(Op):
    op = 'ExpandDims'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': 'Reshape',
            'op': __class__.op,
            'infer': tf_expand_dims_infer,
            'expand_axis': None,
        }, attrs)

    def supported_attrs(self):
        # TODO ugly copying from Reshape op
        return ['axis', ('dim', lambda node: ', '.join(map(str, node['dim']))), 'num_axes']
