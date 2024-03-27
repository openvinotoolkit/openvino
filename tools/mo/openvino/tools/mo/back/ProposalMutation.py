# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.back.ReshapeMutation import ReshapeMutation
from openvino.tools.mo.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.strided_slice import StridedSlice


class ProposalMutation(BackReplacementPattern):
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [ReshapeMutation, StridedSliceMasksNormalizer]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('proposal', {'type': 'Proposal'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['proposal']
        assert len(node.in_ports()) == 3, "Proposal op must have exactly 3 input ports"
        im_info_shape = node.in_port(2).data.get_shape()
        assert im_info_shape is not None

        if np.array_equal(im_info_shape, [1, 6]):
            log.error('The model contains Proposal layer "{}" with input of shape [1, 6]. OpenVINO '
                      'implementation of the Proposal layer uses only 4 first values (indices 0, 1, 2 and 3). '
                      'Elements with indices 4 and 5 will be ignored.'.format(node.soft_get('name', node.id)),
                      extra={'is_warning': True})

            cropped_im_info = create_op_with_const_inputs(graph, StridedSlice, {1: mo_array([0, 0], dtype=np.int32),
                                                                                2: mo_array([1, 3], dtype=np.int32),
                                                                                3: mo_array([1, 1], dtype=np.int32)},
                                                          {'name': 'cropped_im_info',
                                                           'begin_mask': int64_array([1, 1]),
                                                           'end_mask': int64_array([1, 1]),
                                                           'new_axis_mask': int64_array([0, 0]),
                                                           'shrink_axis_mask': int64_array([0, 0]),
                                                           'ellipsis_mask': int64_array([0, 0]),
                                                           'override_output_shape': True,
                                                           })

            node.in_port(2).get_connection().insert_node(cropped_im_info)

            # update the im_info_shape so the next 'if' statement become true
            im_info_shape = int64_array([1, 3])

        if np.array_equal(im_info_shape, [1, 3]) or np.array_equal(im_info_shape, [1, 4]):
            reshape = create_op_node_with_second_input(graph, Reshape, [im_info_shape[1]], {'name': 'im_info/Reshape'})
            node.in_port(2).get_connection().set_destination(reshape.in_port(0))
            reshape.out_port(0).connect(node.in_port(2))
