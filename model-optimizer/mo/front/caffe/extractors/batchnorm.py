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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.common.partial_infer.elemental import copy_shape_infer


def batch_norm_ext(pb_layer, pb_model):
    """
    Extracts properties of the BatchNorm layer.
    In case of scale, scale is merged into mean and variance
    Args:
        pl: proto layer, contains own properties of the layer, i.e epsilon
        ml: caffemodel layer, contains blobs with 0: mean, 1: variance, (opt)2: scale

    Returns:
        attrs object with type, partial inference function and mean/variance properties.
    """
    assert pb_layer, 'Protobuf layer can not be empty'
    param = pb_layer.batch_norm_param
    attrs = {
        'op': 'BatchNormalization',
        'type': 'BatchNormalization',
        'epsilon': param.eps,
        'infer': copy_shape_infer
    }

    if not pb_model:
        return attrs

    blobs = pb_model.blobs
    assert len(blobs) >= 2, 'BatchNorm accepts not less then two input blobs'
    mean = np.array(blobs[0].data)
    variance = np.array(blobs[1].data)

    if len(blobs) == 3:
        scale = blobs[2].data[0]
        if scale != 0:
            scale = 1.0 / scale
        mean *= scale
        variance *= scale

    embed_input(attrs, 1, 'mean', mean, 'biases')
    embed_input(attrs, 2, 'variance', variance, 'weights')

    return attrs
