# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_insert
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.fusing.helpers import get_tensor_in_port, get_value_in_port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class EltwiseChecker(MiddleReplacementPattern):
    """
    Checks if element-wise operation can be converted to ScaleShift or not:
        decision gets made by verifying constant input value shape is like 1,N,1,1
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.EltwiseInputReshape import Eltwise1DInputReshape
        return [Eltwise1DInputReshape]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    @staticmethod
    def set_flags_to_false(node: Node, flags: list):
        for flag in flags:
            node[flag] = False

    def mark_eltwise_node(self, node, feature_channel=None):
        tensor_port, value_port = get_tensor_in_port(node), get_value_in_port(node)
        if tensor_port is None or value_port is None:
            self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
            return

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        if len(connected_in_ports) != 2:
            return

        tensor_shape = tensor_port.data.get_shape()
        out_shape = node.out_port(0).data.get_shape()
        assert tensor_shape is not None and out_shape is not None
        if not np.array_equal(tensor_shape, out_shape):
            # ScaleShift operation doesn't support broadcasting
            self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
            return

        value_shape = value_port.data.get_shape()
        assert value_shape is not None
        assert len(value_shape) <= len(tensor_shape), \
            "No broadcasting was done for elementwise node {} due to previous checks in EltwiseChecker class. " \
            "But constant input rank is larger than tensor input rank, that is inconsistent".format(node.name)

        # if both tensors are 0D they cannot be converted to scaleshift
        if len(tensor_shape) == 0 and len(value_shape) == 0:
            self.set_flags_to_false(node, ['can_be_scaleshift'])
            return

        broadcasted_value_shape = shape_insert(value_shape, 0, [1] * (len(tensor_shape) - len(value_shape)))

        feature_dim = min(1, tensor_shape.size - 1) if node.graph.graph['layout'] == 'NCHW' else -1
        if feature_channel is not None:
            feature_dim = feature_channel
        ones = np.ones(len(tensor_shape), dtype=np.float32)
        possible_shape = ones.copy()
        np.put(possible_shape, feature_dim, tensor_shape.item(feature_dim))

        if not np.array_equal(broadcasted_value_shape, ones) and \
                not np.array_equal(broadcasted_value_shape, possible_shape):
            # ScaleShift weights should have [1,C,1,1]-like or [1,1,1,1]-like shape
            self.set_flags_to_false(node, ['can_be_fused', 'can_be_scaleshift'])
            return

        if len(tensor_shape) not in [2, 4, 5]:
            # ScaleShift operation is supported for 2D, 4D or 5D tensor inputs
            self.set_flags_to_false(node, ['can_be_scaleshift'])
            return

    def find_and_replace_pattern(self, graph: Graph, feature_channel=None):
        for node in graph.get_op_nodes(is_eltwise=True):
            self.mark_eltwise_node(node)
