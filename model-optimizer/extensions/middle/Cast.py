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

    def pattern(self):
        return dict(
            nodes=[('op', dict(op='Cast', dst_type=np.float32))],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        # resulting network is fully floating point, so casts to float are useless
        match['op']['identity'] = True
    