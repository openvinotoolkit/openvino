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

from mo.ops.op import Op
from mo.front.common.partial_infer.const import tf_const_infer


class Const(Op):
    """
    Operation producing constant value stored in the attribute 'value' of shape 'shape'.
    """
    op = 'Const'

    def __init__(self, graph, attrs: dict = None):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'infer': tf_const_infer,
            'value': None,
            'shape': None,
            'data_type': None,
            'out_ports_count': 1,
        }, attrs)
        if not isinstance(self.attrs['value'], np.ndarray):
            self.attrs['value'] = np.array(self.attrs['value'])
        self.attrs['shape'] = np.array(self.attrs['value'].shape, dtype=np.int64)

    def supported_attrs(self):
        return [
            'offset',
            'size'
        ]