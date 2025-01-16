# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.layout import get_features_dim, shape_for_layout
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array, \
    undefined_shape_of_rank, set_input_shapes, compatible_dims
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class ROIAlign(Op):
    op = 'ROIAlign'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        assert 'mode' in attrs, '`mode` attribute is not set for ROIAlign during creation'
        assert 'pooled_h' in attrs, '`pooled_h` attribute is not set for ROIAlign during creation'
        assert 'pooled_w' in attrs, '`pooled_w` attribute is not set for ROIAlign during creation'
        assert 'sampling_ratio' in attrs, '`sampling_ratio` attribute is not set for ROIAlign during creation'
        assert 'spatial_scale' in attrs, '`spatial_scale` attribute is not set for ROIAlign during creation'
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset9',

            'infer': self.infer,
            'reverse_infer': self.reverse_infer,

            'in_ports_count': 3,
            'out_ports_count': 1,

            'aligned_mode': 'asymmetric',
        }, attrs)

    def backend_attrs(self):
        version = self.get_opset()
        if version == 'opset3':
            return [
                ('mode', lambda node: str(node.mode)),
                ('pooled_h', lambda node: str(int(node.pooled_h))),
                ('pooled_w', lambda node: str(int(node.pooled_w))),
                ('sampling_ratio', lambda node: str(int(node.sampling_ratio))),
                ('spatial_scale', lambda node: str(float(node.spatial_scale))),
            ]
        elif version == 'opset9':
            return [
                ('mode', lambda node: str(node.mode)),
                ('pooled_h', lambda node: str(int(node.pooled_h))),
                ('pooled_w', lambda node: str(int(node.pooled_w))),
                ('sampling_ratio', lambda node: str(int(node.sampling_ratio))),
                ('spatial_scale', lambda node: str(float(node.spatial_scale))),
                ('aligned_mode', lambda node: str(node.aligned_mode))
            ]

    @staticmethod
    def infer(node):

        layout = node.graph.graph['layout']
        node_name = node.soft_get('name', node.id)
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 3, \
            'The node "{}" must 3 inputs'.format(node_name)

        assert node.has_valid('pooled_w'), '"pooled_w" attribute is not set for node "{}"'.format(node_name)
        assert node.has_valid('pooled_h'), '"pooled_h" attribute is not set for node "{}"'.format(node_name)
        assert node.has_valid('mode'), '"mode" attribute is not set for node "{}"'.format(node_name)
        assert node.mode in ['avg', 'max'], \
            '"mode" attribute range of values is ["avg", "max"], got {} for node "{}"'.format(node.mode, node_name)
        if node.get_opset() == 'opset9':
            assert node.aligned_mode in ['asymmetric', 'half_pixel_for_nn', 'half_pixel'], \
                '"aligned_mode" attribute range of values is ["asymmetric", "half_pixel_for_nn", "half_pixel"]'
        input_shape = node.in_port(0).data.get_shape()
        rois_shape = node.in_port(1).data.get_shape()
        indices_shape = node.in_port(2).data.get_shape()
        assert input_shape is not None and rois_shape is not None and indices_shape is not None, \
            'The node "{}" input shape is None'.format(node_name)
        assert compatible_dims(rois_shape[0], indices_shape[0]), 'The number of batch indices does not correspond ' \
                                                                 'to number of ROIs for node "{}"'.format(node_name)
        assert compatible_dims(rois_shape[1], 4), 'The size of ROI element must be 4 for node "{}"'.format(node_name)
        assert len(input_shape) == 4, 'The rank of port 0 input tensor of node "{}" must be 4.'.format(node_name)
        node.out_port(0).data.set_shape(
            shape_for_layout(layout,
                             batch=rois_shape[0],
                             features=input_shape[get_features_dim(layout, 4)],
                             height=node.pooled_h,
                             width=node.pooled_w)
        )

    @staticmethod
    def reverse_infer(node):
        set_input_shapes(node,
                         undefined_shape_of_rank(4),
                         shape_array([dynamic_dimension_value, 4]),
                         undefined_shape_of_rank(1))
