# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Dict

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import add_constant_to_negative_values, create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, Node, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.transpose import Transpose


class SSliceComplex(MiddleReplacementPattern):
    """
    Some TF models contain the sub-graph
               SomeOp
                 |
    --------------------------
    |                        |
    StridedSlice          StridedSlice
    |                       |
    ------------------------
         Complex
          |
          |       other inputs
          |       |  ...  |
         -------------------
                 SomeOp1

    Here SomeOp is some node with the real output and with the shape [N_0, ..., N_{k - 1}, 2, N_{k +1}, ..., N_{r - 1}],
    and StridedSlice nodes have output shapes [N_0, ..., N_{k - 1}, N_{k +1}, ..., N_{r - 1}].

    But MO and OpenVINO do not support complex tensors. Hence, we need to replace this sub-graph with.
    1. If k == r - 1, then the replacement should be the subgraph

         SomeOp   other inputs
          |       |  ...  |
         -------------------
                 SomeOp1

    2. In the other case, that is if 0 <= k and k < r - 1 the replacement should be the subgraph

         SomeOp
           |
       Transpose -- input_order
          |
          |
          |   other inputs
          |       |  ...  |
         -------------------
                 SomeOp1

    where the input_order is a Constant, and the value of input_order is [0, ..., k - 1, k + 1, ..., r - 1, k].
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(kind='op', op='StridedSlice')),
                ('strided_slice_real_data', dict(kind='data')),
                ('strided_slice_imag', dict(kind='op', op='StridedSlice')),
                ('strided_slice_imag_data', dict(kind='data')),
                ('complex', dict(op='Complex')),
            ],
            edges=[
                ('strided_slice_real', 'strided_slice_real_data'),
                ('strided_slice_imag', 'strided_slice_imag_data'),
                ('strided_slice_real_data', 'complex', {'in': 0}),
                ('strided_slice_imag_data', 'complex', {'in': 1}),
            ])

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']

        real_input = strided_slice_real.in_port(0).get_source().node
        imag_input = strided_slice_imag.in_port(0).get_source().node
        if real_input.id != imag_input.id:
            log.debug('The pattern does not correspond to operation for complex tensor. Different inputs.')
            return

        real_slices = np.array(strided_slice_real.slices)
        imag_slices = np.array(strided_slice_imag.slices)

        zeros_in_real_input_slices = np.argwhere(real_slices==0).flatten()
        ones_in_imag_input_slices = np.argwhere(imag_slices==1).flatten()

        if len(zeros_in_real_input_slices) != 1 or len(ones_in_imag_input_slices) != 1:
            return

        slice_dim_for_real_part = zeros_in_real_input_slices[0]
        slice_dim_for_imag_part = ones_in_imag_input_slices[0]
        if slice_dim_for_real_part != slice_dim_for_imag_part:
            return

        emulated_complex_tensor_shape = strided_slice_real.in_port(0).data.get_shape()
        if emulated_complex_tensor_shape is None:
            return

        emulated_complex_tensor_rank = len(emulated_complex_tensor_shape)
        complex_node = match['complex']

        for dst in complex_node.out_port(0).get_connection().get_destinations():
            after_complex_node = dst.node
            # TODO: now it does not support adjustment of `axis` inputs for other operations such Gather, Concat, etc.
            # It does not traverse the full path affected by complex numbers for adjusting the corresponding operations.
            # It can affect other models with complex numbers for which we can generate incorrect IRs or offline transformation fails.
            if after_complex_node.type == 'Roll':
                add_constant_to_negative_values(after_complex_node, 2, int64_array(emulated_complex_tensor_rank))

        input_slices_have_ellipsis = len(np.argwhere(real_slices == Ellipsis).flatten()) != 0

        # If output of SomeOp is sliced on the last dimension on the last dimension (like described in 1 case), skipping Complex op is enough.
        # Otherwise, (like described in 2 case) Transpose insertion is needed to align data arrangement.
        if slice_dim_for_real_part == emulated_complex_tensor_rank - 1 or input_slices_have_ellipsis:
            complex_node.out_port(0).get_connection().set_source(strided_slice_real.in_port(0).get_source())
        else:
            complex_node_name = complex_node.soft_get('name', complex_node.id)
            perm = int64_array([*range(0, slice_dim_for_real_part),
                                *range(slice_dim_for_real_part + 1, emulated_complex_tensor_rank),
                                slice_dim_for_real_part])
            transpose = create_op_with_const_inputs(graph, Transpose, {1: perm},
                                                    {'name': complex_node_name + '/cmplx'})
            complex_node.out_port(0).get_connection().set_source(transpose.out_port(0))
            strided_slice_real.in_port(0).get_source().connect(transpose.in_port(0))
            rename_nodes([(complex_node, complex_node_name + '/to_be_removed'), (transpose, complex_node_name)])
