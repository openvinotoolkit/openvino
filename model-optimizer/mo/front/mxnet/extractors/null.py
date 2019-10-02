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

from mo.front.common.partial_infer.const import tf_const_infer
from mo.front.common.partial_infer.elemental import single_output_infer


def null_ext(attr_dict):
    if 'value' in attr_dict:
        value = attr_dict['value']
        return {
            'op': 'Const',
            'value': value,
            'shape': np.array(value.shape, dtype=np.int64),
            'infer': tf_const_infer
        }
    else:
        return {
            'op': 'Parameter',
            'type': 'Parameter',
            'shape': None,
            'infer': lambda node: single_output_infer(node, lambda n: n.shape)
        }
