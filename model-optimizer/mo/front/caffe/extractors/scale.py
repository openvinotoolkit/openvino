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

from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.utils import NamedAttrsClass


def scale_ext(pl, ml):
    param = pl.scale_param
    attrs = {
        'op': 'ScaleShift',
        'type': 'ScaleShift',
        'infer': copy_shape_infer
    }
    if ml is None:
        # default weights and biases for scale layer if the caffemodel file doesn't contain them
        ml = NamedAttrsClass({'blobs': np.array([NamedAttrsClass({'data': np.array([1])}),
                                                 NamedAttrsClass({'data': np.array([0])})])})
    attrs.update(weights_biases(param.bias_term, ml))
    return attrs
