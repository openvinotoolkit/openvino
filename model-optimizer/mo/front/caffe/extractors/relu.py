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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.elemental import copy_shape_infer


def relu_ext(proto_layer, model_layer):
    assert proto_layer, 'Protobuf layer can not be empty'
    param = proto_layer.relu_param
    negative_slope = param.negative_slope

    attrs = {
        'op': 'ReLU',
        'type': 'ReLU',
        'negative_slope': negative_slope,
        'infer': copy_shape_infer
    }
    attrs.update(layout_attrs())
    return attrs
