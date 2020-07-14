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

from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const


def correct_pad(pad):
    return int64_array([pad] if not isinstance(pad, list) else pad)


class InterpolateV1ToInterpolateV4(FrontReplacementSubgraph):
    """
    This transformation replaces the layer Interpolate-1 with the layer Interpolate-3.
    """
    enabled = True

    def run_after(self):
        from extensions.front.InterpolateNormalizer import InterpolateNormalizer
        return [InterpolateNormalizer]

    def pattern(self):
        return dict(
            nodes=[('op', {'type': 'Interpolate'})],
            edges=[],
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']
        opset = node.get_opset()
        if opset != 'opset1':
            return

        transformation_mode = 'align_corners' if int(node.soft_get('align_corners', 0)) else 'half_pixel'
        input_node = node.in_port(0).get_source().node
        sizes = node.in_port(1).get_source().node
        axes_node = Const(graph, {'name': node.name + '/axis_', 'value': int64_array(node.axes)}).create_node()
        interpolate4 = Interpolate(graph,
                                   {
                                       'mode': node.mode,
                                       'antialias': node.antialias,
                                       'coordinate_transformation_mode': transformation_mode,
                                       'pads_begin': correct_pad(node.soft_get('pads_begin', 0)),
                                       'pads_end': correct_pad(node.soft_get('pads_end', 0)),
                                       'nearest_mode': 'round_prefer_floor',
                                       'cube_coeff': -0.75,
                                       'version': 'opset4',
                                       'in_ports_count': 3,
                                   }).create_node([input_node, sizes, axes_node])
        node.replace_node(interpolate4)
