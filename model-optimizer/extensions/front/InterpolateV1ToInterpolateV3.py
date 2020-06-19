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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph


class InterpolateV1ToInterpolateV3(FrontReplacementOp):
    op = 'Interpolate'
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        opset = node.get_opset()
        if opset != 'opset1':
            return

        if int(node.soft_get('align_corners', default=0)):
            node['coordinate_transformation_mode'] = 'align_corners'
            del node['align_corners']
        else:
            node['coordinate_transformation_mode'] = 'half_pixel'

        node['nearest_mode'] = 'round_prefer_floor'
        node['cube_coeff'] = -0.75

        pads_begin = node.soft_get('pads_begin', 0)
        pads_end = node.soft_get('pads_end', 0)

        node['pads_begin'] = int64_array([pads_begin] if not isinstance(pads_begin, list) else pads_begin)
        node['pads_end'] = int64_array([pads_end] if not isinstance(pads_end, list) else pads_end)

        node.version = 'opset3'
