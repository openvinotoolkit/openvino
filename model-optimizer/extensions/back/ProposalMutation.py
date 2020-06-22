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
import logging as log

import numpy as np

from extensions.back.ReshapeMutation import ReshapeMutation
from extensions.back.StridedSliceMasksNormalizer import StridedSliceMasksNormalizer
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.strided_slice import StridedSlice


class ProposalMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True

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
            log.error('The model contains Proposal layer "{}" with input of shape [1, 6]. Inference Engine '
                      'implementation of the Proposal layer uses only 4 first values (indices 0, 1, 2 and 3). '
                      'Elements with indices 4 and 5 will be ignored.'.format(node.soft_get('name', node.id)),
                      extra={'is_warning': True})

            cropped_im_info = create_op_with_const_inputs(graph, StridedSlice, {1: np.array([0, 0], dtype=np.int32),
                                                                                2: np.array([1, 3], dtype=np.int32),
                                                                                3: np.array([1, 1], dtype=np.int32)},
                                                          {'name': 'cropped_im_info',
                                                           'begin_mask': int64_array([1, 1]),
                                                           'end_mask': int64_array([1, 1]),
                                                           'new_axis_mask': int64_array([0]),
                                                           'shrink_axis_mask': int64_array([0]),
                                                           'ellipsis_mask': int64_array([0]),
                                                           'override_output_shape': True,
                                                           })

            node.in_port(2).get_connection().insert_node(cropped_im_info)

            # update the im_info_shape so the next 'if' statement become true
            im_info_shape = int64_array([1, 3])

        if np.array_equal(im_info_shape, [1, 3]) or np.array_equal(im_info_shape, [1, 4]):
            reshape = Reshape(graph, dict(name="im_info/Reshape")).create_node()
            const = Const(graph, dict(value=[im_info_shape[1]])).create_node()
            node.in_port(2).get_connection().set_destination(reshape.in_port(0))
            const.out_port(0).connect(reshape.in_port(1))
            reshape.out_port(0).connect(node.in_port(2))

        if node.has_port('out', 1) and not node.out_port(1).disconnected():
            # This is the case when Proposal layer is used from extension, not from opset.
            # Setting version attribute is not recommended, this will be fixed after Proposal will be updated in IE.
            graph.node[node.id]['version'] = 'extension'
