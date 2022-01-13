# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze(MiddleReplacementPattern):
    r"""
    The transformation removes shrink_axis and new_axis masks from StridedSlice and inserts Squeeze/Unsqueeze nodes
    instead of these masks.
    This is necessary if StridedSlice is to be executed in original N(D)HWC layout, because
    the operation does not have reinterp_shape attribute and MO can not insert NC(D)HW -> N(D)HWC Transpose in
    openvino/tools/mo/middle/InsertLayoutPropagationTransposes.py.
    """
    enabled = True
    force_shape_inference = True


    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_before(self):
        return [InsertLayoutPropagationTranspose]

    def run_after(self):
        from openvino.tools.mo.middle.StridedSliceNormalizer import StridedSliceNormalizer
        from openvino.tools.mo.middle.ConvertGroupedStridedSlice import ConvertGroupedStridedSlice
        return [StridedSliceNormalizer, ConvertGroupedStridedSlice]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='StridedSlice'):
            input_shape = node.in_port(0).data.get_shape()

            shrink_axis_mask = node.soft_get('shrink_axis_mask', np.zeros(len(input_shape)))
            new_axis_mask = node.soft_get('new_axis_mask', np.zeros(len(input_shape)))

            squeeze_indices = np.nonzero(shrink_axis_mask)[0]
            squeeze_indices_original = squeeze_indices.copy()
            unsqueeze_indices = np.nonzero(new_axis_mask)[0]

            # assert begin ends are correct for squeeze

            latest_node = node
            if len(squeeze_indices) > 0:
                for i, axis in enumerate(squeeze_indices):
                    squeeze_indices[i] -= np.count_nonzero(unsqueeze_indices < squeeze_indices[i])

                squeeze_node = create_op_node_with_second_input(node.graph, Squeeze, int64_array(squeeze_indices))
                node.out_port(0).get_connection().insert_node(squeeze_node)
                node.shrink_axis_mask = np.zeros(len(input_shape))
                latest_node = squeeze_node
            if len(unsqueeze_indices) > 0:
                for mask in StridedSlice.get_mask_names():
                    node[mask] = np.delete(int64_array(node[mask]), unsqueeze_indices)

                slice_rank = node.in_port(1).data.get_shape()[0]
                slice_along_axes = list(set(range(0, slice_rank)) - set(unsqueeze_indices))

                gather_begin = create_op_with_const_inputs(graph, Gather, {1: slice_along_axes, 2: int64_array(0)})
                node.in_port(1).get_connection().insert_node(gather_begin)

                gather_end = create_op_with_const_inputs(graph, Gather, {1: slice_along_axes, 2: int64_array(0)})
                node.in_port(2).get_connection().insert_node(gather_end)

                if node.is_in_port_connected(3):
                    gather_strides = create_op_with_const_inputs(graph, Gather,
                                                                 {1: slice_along_axes, 2: int64_array(0)})
                    node.in_port(3).get_connection().insert_node(gather_strides)

                for i, axis in enumerate(unsqueeze_indices):
                    unsqueeze_indices[i] -= np.count_nonzero(squeeze_indices_original < unsqueeze_indices[i])

                unsqueeze_node = create_op_node_with_second_input(node.graph, Unsqueeze, int64_array(unsqueeze_indices))
                latest_node.out_port(0).get_connection().insert_node(unsqueeze_node)

            node['need_shape_inference'] = True
            node['override_output_shape'] = True
            node.infer(node)
