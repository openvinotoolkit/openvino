# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.lrn import LRN


class LRN_normalization(BackReplacementPattern):
    """
    Transforming LRN with `region` attribute to LRN with second `axis`-input
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('lrn', dict(kind='op', op='AttributedLRN'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['lrn']
        name = node.soft_get('name', node.id)
        assert node.has_valid('region')
        assert node.region in ['across', 'same']

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None
        axis = [1] if node.region == 'across' else list(range(2, input_shape.size))

        new_lrn = create_op_node_with_second_input(graph, LRN, int64_array(axis), {
            'name': name,
            'alpha': node.alpha,
            'beta': node.beta,
            'size': node.local_size,
            'bias': node.bias,
        })

        node.out_port(0).get_connection().set_source(new_lrn.out_port(0))
        node.in_port(0).get_connection().set_destination(new_lrn.in_port(0))
