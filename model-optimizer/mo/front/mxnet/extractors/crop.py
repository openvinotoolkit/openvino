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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.crop import crop_infer


def crop_ext(attr):
    offset = attr.tuple("offset", int, ())
    axis = attr.int("num_args", 0)

    node_attrs = {
        'op': 'Crop',
        'type': 'Crop',
        'axis': axis,
        'offset': list(offset),
        'dim': None,
        'infer': crop_infer
    }
    node_attrs.update(layout_attrs())
    return node_attrs
