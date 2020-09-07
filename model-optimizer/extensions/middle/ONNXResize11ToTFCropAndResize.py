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


import logging as log
import math

import numpy as np

from mo.front.common.layout import get_height_dim, get_width_dim
from mo.front.common.partial_infer.utils import float_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Node, Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.roipooling import ROIPooling


def replace_resize_with_tf_crop_and_resize(graph: Graph, resize: Node):
    log.debug("Converting of ONNX Resize-11 with coordinate_transformation_node == 'tf_crop_and_resize' to "
              "ROIPooling + ONNXResize11 is triggered for node {}.".format(resize.soft_get('name', resize.id)))

    input_shape = resize.in_port(0).data.get_shape()
    input_rank = len(input_shape)
    resize_name = resize.soft_get('name', resize.id)
    if input_rank != 4:
        log.warning('The input shape is not 4D for op with name {}'.format(resize_name))
        return

    roi = resize.in_port(1).data.get_value()
    assert roi is not None, "The input 'roi' is None for ONNXResize11 operation with name {}".format(resize_name)

    roi_len = len(roi)

    assert roi_len == 2 * input_rank, \
        "Incorrect value of 'roi' input for ONNXResize11 operation with name {}".format(resize_name)

    roi_starts = roi[:input_rank]
    roi_ends = roi[input_rank:]

    assert np.array_equal(roi_starts, np.zeros(input_rank)), "Non-zero roi starts are not supported"

    layout = graph.graph['layout']
    height_dim = get_height_dim(layout, input_rank)
    width_dim = get_width_dim(layout, input_rank)

    roi_x_end, roi_y_end = roi_ends[width_dim], roi_ends[height_dim]

    pooling_rois = float_array([0, 0, 0, roi_x_end, roi_y_end]).reshape((1, 5))
    pooling_attributes = {'name': resize_name + '/ROIPooling_',
                          'spatial_scale': 1.0,
                          'method': 'bilinear',
                          'pooled_h': math.floor(roi_y_end * input_shape[height_dim]),
                          'pooled_w': math.floor(roi_x_end * input_shape[width_dim]),
                          }

    resize.coordinate_transformation_mode = 'align_corners'
    roi_pooling_node = create_op_with_const_inputs(graph, ROIPooling, {1: pooling_rois}, pooling_attributes)

    connection_of_resize_input = resize.in_port(0).get_connection()
    connection_of_resize_input.set_destination(roi_pooling_node.in_port(0))

    roi_pooling_node.out_port(0).connect(resize.in_port(0))


class ONNXResiz11WithTFCropAndResizeToROIPoolingAndONNXResize(MiddleReplacementPattern):
    """
    The transformation replaces ONNX Resize 11 (coordinate_transformation_node == 'tf_crop_and_resize') with
    ROIPooling + ONNXResize11.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.ONNXResize11ToInterpolateV4 import ONNXResize11ToInterpolate4
        return [ONNXResize11ToInterpolate4]

    def find_and_replace_pattern(self, graph: Graph):
        resize11_ops = graph.get_op_nodes(op='ONNXResize11', coordinate_transformation_node='tf_crop_and_resize')
        for resize in resize11_ops:
            replace_resize_with_tf_crop_and_resize(graph, resize)
