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

from mo.front.common.partial_infer.roipooling import roipooling_infer


def roipooling_ext(proto_layer, model_layer):
    param = proto_layer.roi_pooling_param
    return {
        'type': 'ROIPooling',
        'pooled_h': param.pooled_h,
        'pooled_w': param.pooled_w,
        'spatial_scale': param.spatial_scale,
        'infer': roipooling_infer
    }
