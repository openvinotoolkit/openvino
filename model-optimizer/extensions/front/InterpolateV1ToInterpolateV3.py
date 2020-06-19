"""
 Copyright (C) 2020 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph


class InterpolateV1ToInterpolateV3(FrontReplacementOp):
    op = 'Interpolate'
    enabled = False

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        opset = node.get_opset()
        if opset != 'opset1':
            return