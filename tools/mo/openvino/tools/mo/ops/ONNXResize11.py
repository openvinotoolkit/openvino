# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class ONNXResize11Op(Op):
    op = 'ONNXResize11'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'infer': ONNXResize11Op.onnx_resize_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'coordinate_transformation_mode',
            'cube_coeff',
            'exclude_outside',
            'extrapolation_value',
            'mode',
            'nearest_mode'
        ]

    @staticmethod
    def onnx_resize_infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        assert (node.is_in_port_connected(0) and (node.is_in_port_connected(2) or node.is_in_port_connected(3))), \
            "One of the scales or sizes inputs must be connected to Node {} with op {}." \
            "".format(node.soft_get("name", node.id), node.op)

        assert node.coordinate_transformation_mode != 'tf_crop_and_resize', \
            'Mode tf_crop_and_resize is not supported for op {} with name {}'.format(node.op,
                                                                                     node.soft_get("name", node.id))

        if not node.is_in_port_connected(3):
            # i.e. input 'sizes' is not given
            input2_value = node.in_port(2).data.get_value()
            assert input2_value is not None, \
                "Node {} with op {} has no value in input port 2".format(node.soft_get('name', node.id), node.op)
            scale = mo_array(input2_value)
            output_shape = np.floor(input_shape * scale + 1.0e-6).astype(np.int64)
        else:
            # i.e. input 'sizes' is given
            sizes = node.in_port(3).data.get_value()
            assert sizes is not None, \
                "Node {} with op {} has no value in input port 3".format(node.soft_get("name", node.id), node.op)
            output_shape = input_shape.copy()
            spatial_dimension_indices = range(2, len(input_shape))
            output_shape[spatial_dimension_indices] = sizes[2:]

        node.out_port(0).data.set_shape(output_shape)
