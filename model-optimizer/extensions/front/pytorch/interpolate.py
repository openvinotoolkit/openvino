"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.ops.interpolate import Interpolate
from extensions.ops.upsample import UpsampleOp
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const

class InterpolateReplacer(FrontReplacementOp):
    op = 'Upsample'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        mode = node.module.mode
        if mode == 'bilinear':
            mode = 'linear'
        align_corners = node.module.align_corners

        if mode == 'linear' and not align_corners:
            height = node.module.size[0]
            width = node.module.size[1]
            attrs = {
                'name': node.name,
                'version': 'opset4',
                'height': height,
                'width': width,
                'mode': mode,
                'axes': [2, 3],
                'pads_begin': [0, 0],
                'pads_end': [0, 0],
                'align_corners': node.module.align_corners,
                'shape_calculation_mode': 'sizes',
            }

            sizes = Const(graph, {'value': np.array([height, width])}).create_node()
            axes = Const(graph, {'value': np.array([2, 3])}).create_node()
            scales = Const(graph, {'value': np.array([1, 1], dtype=np.float32)}).create_node()
            interp = Interpolate(graph, attrs).create_node([node.in_node(0), sizes, scales, axes])
        else:
            if node.module.size:
                attrs = {
                    'name': node.name,
                    'version': 'opset1',
                    'height': node.module.size[0],
                    'width': node.module.size[1],
                    'mode': mode,
                    'axes': [2, 3],
                    'align_corners': node.module.align_corners,
                }
                interp = Interpolate(graph, attrs).create_node([node.in_node(0)])
            else:
                assert(node.module.scale_factor)
                attrs = {
                    'name': node.name,
                    'height_scale': node.module.scale_factor,
                    'width_scale': node.module.scale_factor,
                    'mode': mode,
                    'align_corners': node.module.align_corners,
                }
                interp = UpsampleOp(graph, attrs).create_node([node.in_node(0)])

        return [interp.id]
