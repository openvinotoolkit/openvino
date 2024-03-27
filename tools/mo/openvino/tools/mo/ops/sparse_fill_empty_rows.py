# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, is_fully_defined
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class SparseFillEmptyRows(Op):
    """
    The operation fills empty rows in the input 2-D sparse tensor with a default value.
    For more details see https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-fill-empty-rows

    4 inputs:
        - [0, required] input indices of the sparse tensor (2D),
        - [1, required] input values of the sparse tensor (1D),
        - [2, required] shape of the sparse tensor. Value of this input is required for the Model Optimizer (1D),
        - [3, required] default value to insert at rows missing from the input sparse tensor (0D),

    3 outputs:
        - [0, optional] indices of the filled sparse tensor (2D)
        - [1, optional] values of the filled sparse tensor (1D)
        - [2, optional] indicator of whether the dense row was missing in the input sparse tensor (1D)
    """
    op = 'SparseFillEmptyRows'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'version': 'experimental',
            'infer': self.infer,
            'in_ports_count': 4,
            'out_ports_count': 3
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 4

        # check that shape value is defined that is needed for shape inference
        shape = node.in_node(2)
        assert shape.value is not None and shape.value.size == 2, \
            "SparseFillEmptyRows is supported only with constant shape value"

        shape_value = int64_array(shape.value)

        # check that default value is scalar
        default_value = node.in_node(3)
        assert default_value.shape is not None and len(default_value.shape) == 0, \
            "Default value for SparseFillEmptyRows must be scalar"

        if node.is_out_port_connected(0):  # set a shape for output indices
            if is_fully_defined(shape_value):
                node.out_port(0).data.set_shape([np.prod(shape_value), 2])
            else:
                node.out_port(0).data.set_shape([dynamic_dimension_value, 2])
        if node.is_out_port_connected(1):  # set a shape for output values
            if is_fully_defined(shape_value):
                node.out_port(1).data.set_shape([np.prod(shape_value)])
            else:
                node.out_port(1).data.set_shape([dynamic_dimension_value])
        if node.is_out_port_connected(2):  # set a shape for empty row indicator
            node.out_port(2).data.set_shape([shape_value[0]])
