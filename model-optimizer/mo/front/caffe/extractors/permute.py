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

import numpy as np
from mo.front.common.partial_infer.transpose import transpose_infer


def permute_ext(pb_layer, pb_model):
    param = pb_layer.permute_param
    return {
        'type': 'Permute',
        'op': 'Permute',
        'order': np.array(param.order),
        'infer': transpose_infer
    }
