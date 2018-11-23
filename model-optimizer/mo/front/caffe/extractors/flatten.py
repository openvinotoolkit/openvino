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

from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.partial_infer.flatten import flatten_infer


def flatten_ext(pl, ml):
    param = pl.flatten_param
    update_attrs = {
        'axis': param.axis,
        'end_axis': param.end_axis,
        'num_axes': 0
    }
    mapping_rule = merge_attrs(param, update_attrs)
    mapping_rule.update({
        'type': "Flatten",
        'infer': flatten_infer
    })
    return mapping_rule
