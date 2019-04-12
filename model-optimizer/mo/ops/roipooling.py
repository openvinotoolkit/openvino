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
from mo.ops.op import Op


class ROIPooling(Op):
    op = 'ROIPooling'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'pooled_h': None,
            'pooled_w': None,
            'spatial_scale': 0.0625,
            'type': __class__.op,
            'op': __class__.op,
            'infer': roipooling_infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['pooled_h', 'pooled_w', 'spatial_scale', 'method']
