# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


def correct_pad(pad):
    return int64_array([pad] if not isinstance(pad, list) else pad)


class InterpolateV1ToInterpolate(FrontReplacementPattern):
    """
    This transformation replaces the operation Interpolate-1 with the operation Interpolate-4.
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.InterpolateNormalizer import InterpolateNormalizer
        return [InterpolateNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Interpolate', version='opset1'):
            transformation_mode = 'align_corners' if int(node.soft_get('align_corners', 0)) else 'half_pixel'
            interpolate1_name = node.soft_get('name', node.id)
            interpolate4 = create_op_with_const_inputs(graph, Interpolate,
                                                       {
                                                           2: mo_array([1.0, 1.0]),
                                                           3: int64_array(node.axes)
                                                       },
                                                       {
                                                           'mode': node.mode,
                                                           'antialias': node.antialias,
                                                           'coordinate_transformation_mode': transformation_mode,
                                                           'pads_begin': correct_pad(node.soft_get('pads_begin', 0)),
                                                           'pads_end': correct_pad(node.soft_get('pads_end', 0)),
                                                           'nearest_mode': 'round_prefer_floor',
                                                           'cube_coeff': -0.75,
                                                           'shape_calculation_mode': 'sizes',
                                                           'version': 'opset4',
                                                           'in_ports_count': 4,
                                                       })

            interpolate1_input_connection = node.in_port(0).get_connection()
            interpolate1_input_connection.set_destination(interpolate4.in_port(0))

            sizes_connection = node.in_port(1).get_connection()
            sizes_connection.set_destination(interpolate4.in_port(1))

            node.out_port(0).get_connection().set_source(interpolate4.out_port(0))
            rename_nodes([(node, interpolate1_name + '/delete'), (interpolate4, interpolate1_name)])
