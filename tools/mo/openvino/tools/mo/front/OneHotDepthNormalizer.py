# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.reshape import Reshape


class OneHotDepthNormalizer(FrontReplacementPattern):
    """
    Transformation performs squeezing one-element tensors on 1st input in OneHot into 0D scalars. This transformation
    allows to avoid problems with some models produced by tf2onnx which have 1D depth in OneHot.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('onehot', dict(kind='op', type='OneHot'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['onehot']
        node_name = node.soft_get('name', node.id)
        reshape = create_op_with_const_inputs(graph, Reshape, {1: int64_array([])}, {'name': node_name + '/Reshape'})
        node.in_port(1).get_connection().insert_node(reshape)
