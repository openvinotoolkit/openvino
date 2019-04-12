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

from mo.front.common.partial_infer.squeeze import tf_squeeze_infer
from mo.ops.op import Op


class Squeeze(Op):
    op = 'Squeeze'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'dim': None,
            'kind': 'op',
            'type': 'Reshape',
            'op': __class__.op,
            'infer': tf_squeeze_infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)
