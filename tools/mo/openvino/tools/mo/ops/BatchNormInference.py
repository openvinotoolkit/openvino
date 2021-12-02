# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
            'reverse_infer': self.reverse_infer
        }, attrs)

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())

    @staticmethod
    def reverse_infer(node):
        output_shape = node.out_port(0).data.get_shape()
        if output_shape is not None and node.in_port(0).data.get_shape() is None:
            node.in_port(0).data.set_shape(output_shape)
