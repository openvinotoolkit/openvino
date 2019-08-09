"""
 Copyright (c) 2019 Intel Corporation

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


class Fill(Op):
    """
    The Fill layer tiles the second input tensor (0D constant) to the shape specified in the first input.

    This operation is converted to Broadcast layer.
    """
    op = 'Fill'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'infer': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)
