"""
 Copyright (c) 2019 Intel Corporation

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
from extensions.back.ShapeOfToShape import ShapeOfToShape
from extensions.ops.interp import InterpOp
from extensions.ops.resample import ResampleOp
from extensions.ops.resize_factor_utils import factor_update
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.shape import Shape
from mo.utils.shape import node_to_get_batch_value, node_to_get_features_dimension_value, \
    new_shape_node_from_shape_nodes


class InterpolateToInterpOrResample(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    type_map = {
        'linear': 'caffe.ResampleParameter.LINEAR',
        'nearest': 'caffe.ResampleParameter.NEAREST',
        'cubic': 'caffe.ResampleParameter.CUBIC',
        'area': 'caffe.ResampleParameter.AREA',
    }

    def run_before(self):
        return [ShapeOfToShape]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('interpolate', {'type': 'Interpolate'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['interpolate']

        assert 1 in node.in_ports() and not node.in_port(1).disconnected() and \
               node.in_port(1).data.get_value() is not None, 'Interpolate node {} is corrupted: no 1-port input found'

        # common
        mode = node.mode
        assert mode in ['linear', 'nearest', 'cubic', 'area']
        in_shape = node.in_port(0).data.get_shape()
        assert in_shape is not None and len(in_shape) == 4
        out_shape = node.out_port(0).data.get_shape()
        assert out_shape is not None and len(out_shape) == 4
        in_height, in_width = in_shape[2], in_shape[3]
        out_height, out_width = out_shape[2], out_shape[3]
        factor = factor_update(
            None if not node.has_valid('factor') else node.factor,
            [float(out_height) / in_height, float(out_width) / in_width],
            [in_height, in_width],
            [out_height, out_width],
            node.soft_get('name')
        )
        update_attrs = {
            'width': out_width,
            'height': out_height,
            'factor': factor,
        }

        if (node.has_valid('shrink_factor') and node.has_valid('zoom_factor')) or factor is None:
            del update_attrs['factor']
            if node.has('factor'):
                del node['factor']

        if ((node.has_valid('shrink_factor') and node.shrink_factor != 1) or
            (node.has_valid('zoom_factor') and node.zoom_factor != 1) or 'factor' in update_attrs) \
                and ((not node.has_valid('width') or node.width == 0) and
                     (not node.has_valid('height') or node.height == 0)):
            update_attrs['width'] = 0
            update_attrs['height'] = 0

        # specific
        if mode in ['nearest', 'cubic', 'area'] or node.has_and_set('convert_to_resample'):
            assert not node.align_corners
            assert node.pads_begin == 0 and node.pads_end == 0
            update_attrs['resample_type'] = InterpolateToInterpOrResample.type_map[mode]
            ResampleOp.update_node_stat(node, update_attrs)

            if not graph.graph['cmd_params'].keep_shape_ops or graph.graph['fw'] != 'tf':
                node.in_port(1).disconnect()
            else:
                # we avoid making resample non-reshapable for tf version
                shape = Shape(graph, {}).create_node()
                node.in_port(0).get_source().connect(shape.in_port(0))

                batch = node_to_get_batch_value(shape)
                features = node_to_get_features_dimension_value(shape)
                full_shape = new_shape_node_from_shape_nodes([batch, features, node.in_port(1).get_source().node])
                node.in_port(1).get_connection().set_source(full_shape.out_port(0))
                full_shape['override_output_shape'] = True

        elif mode == 'linear':
            update_attrs.update({
                'pad_beg': node.pads_begin,
                'pad_end': node.pads_end,
                'align_corners': node.align_corners,
            })
            InterpOp.update_node_stat(node, update_attrs)
            node.in_port(1).disconnect()
