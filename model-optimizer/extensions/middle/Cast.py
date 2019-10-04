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
import logging as log

import numpy as np

from extensions.middle.RemoveIdentity import RemoveIdentity
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class CastToFloatMark(MiddleReplacementPattern):
    enabled = True

    def run_before(self):
        return [RemoveIdentity]

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    identity_list = [np.float32, np.double, np.int32, np.int64, np.uint8, np.bool]

    def pattern(self):
        return dict(
            nodes=[('op', dict(op='Cast', dst_type=lambda dst_type: dst_type in self.identity_list))],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        # resulting network is fully floating point, so casts to float are useless
        node = match['op']
        name = node.soft_get('name', node.id)
        dst_type = node.dst_type

        if node.out_port(0).data.get_value() is None:
            if dst_type in [np.int32, np.int64]:
                log.warning('Deleting Cast node {} to {} from network since Cast operation isn\'t supported yet. '
                            'Inference results can be incorrect'.format(name, dst_type))

            match['op']['identity'] = True
