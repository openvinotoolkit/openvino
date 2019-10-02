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

from mo.front.common.partial_infer.multi_box_prior import multi_box_prior_infer_mxnet
from mo.utils.error import Error


def multi_box_prior_ext(attr):
    min_size = attr.tuple("sizes", float, (1, 1))
    offset_y, offset_x = attr.tuple("offsets", float, (0.5, 0.5))
    clip = 0 if not attr.bool("clip", False) else 1
    aspect_ratio = attr.tuple("ratios", float, None)
    step_y, step_x = attr.tuple("steps", float, (-1, -1))
    if len(aspect_ratio) == 0:
        aspect_ratio = [1.0]

    node_attrs = {
        'type': 'PriorBox',
        'img_size': 0,
        'img_h': 0,
        'img_w': 0,
        'step': step_x,
        'step_h': 0,
        'step_w': 0,
        'offset': offset_x,
        'variance': '0.100000,0.100000,0.200000,0.200000',
        'flip': 0,
        'clip': clip,
        'min_size': min_size,
        'max_size': '',
        'aspect_ratio': list(aspect_ratio),
        'scale_all_sizes': 0,
        'infer': multi_box_prior_infer_mxnet
    }
    return node_attrs
