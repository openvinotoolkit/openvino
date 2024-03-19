# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class BatchNormInference(Op):
    """
    BatchNormInference will be replaced by BNToScaleShift FrontReplacer for Caffe or convert_batch_norm 
    function for other frameworks
    """
    op = 'batchNormInference'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 5,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        }, attrs)

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
