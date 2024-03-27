# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import compatible_shapes
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class SparseSegmentSum(Op):
    ''' The operation computes the sum along sparse segments of a tensor.
        For more details, see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-segment-sum.

        Three inputs:
            - [0, required] Data tensor from which rows are selected for the sum (ND),
            - [1, required] Tensor of indices of selected rows from the first input tensor along 0 dimension (1D),
            - [2, required] Tensor of segment IDs to which selected rows for the sum belong.
                            Selected rows belonging to the same segment are summed up. The tensor has the same size as the second input.
                            Values must be sorted and can be repeated. (1D).
        
        One output:
            - [0, required] The output has the same shape as the data tensor, except for dimension 0, which has a size equal to a number of segments. (ND)
    '''
    op = 'SparseSegmentSum'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'version': 'experimental',
            'infer': __class__.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        # check a number of input/output edges
        assert len(node.in_nodes()) == 3
        assert len(node.out_nodes()) == 1

        data_shape = node.in_port(0).data.get_shape()
        indices_shape = node.in_port(1).data.get_shape()
        segment_ids_shape = node.in_port(2).data.get_shape()
        data_value = node.in_port(0).data.get_value()
        indices_value = node.in_port(1).data.get_value()
        segment_ids_value = node.in_port(2).data.get_value()

        # check input shapes
        assert data_shape is not None, \
            "Shape for input data tensor to SparseSegmentSum must be defined"
        assert indices_shape is not None and indices_shape.size == 1, \
            "SparseSegmentSum supports only 1D indices tensor"
        assert segment_ids_shape is not None and segment_ids_shape.size == 1, \
            "SparseSegmentSum supports only 1D segment IDs tensor"
        assert compatible_shapes(segment_ids_shape, indices_shape), \
            "Indices and segment IDs tensors must have compatible shapes"

        # computes output shape
        output_shape = data_shape
        output_shape[0] = segment_ids_shape[0]
        node.out_port(0).data.set_shape(output_shape)

        # infer if all input is constant
        if data_value is None or indices_value is None or segment_ids_value is None:
            return

        # check that values in segment_ids are sorted
        for i in range(1, len(segment_ids_value)):
            assert segment_ids_value[i-1] <= segment_ids_value[i], \
                "Values in segment IDs are not sorted"
        num_segments = int(segment_ids_value[-1]) + 1

        # check that indices are in a range [0, data_shape[0])
        assert np.all(indices_value >= 0) and np.all(indices_value < data_shape[0]), \
            "Some value in indices tensor is out of range"

        # infer
        output_value = np.zeros([num_segments] + data_shape[1:].tolist(), dtype=np.float32)
        output_shape = output_value.shape
        for i in range(len(segment_ids_value)):
            segment_id = int(segment_ids_value[i])
            indice = int(indices_value[i])
            output_value[segment_id, :] += data_value[indice, :]
        node.out_port(0).data.set_shape(output_shape)
        node.out_port(0).data.set_value(output_value)
