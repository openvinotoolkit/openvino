# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.op import PermuteAttrs


class Deconvolution3rdInputNormalization(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='Deconvolution'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        if not node.has_port('in', 2) or node.in_port(2).disconnected() or not node.has_and_set('shape_input'):
            return

        if node.has_valid('layout') and not node.layout.startswith('NC') and graph.graph['layout'] == 'NCHW':
            input_shape_rank = len(node.in_port(0).data.get_shape())
            permutation = PermuteAttrs.get_nhwc_to_nchw_permutation(input_shape_rank)

            data_node = node.in_node(2)

            name = node.soft_get('name', node.id) + '/ShapeGather'
            const = Const(graph, {'value': permutation.perm, 'name': name + '/Const',
                                  'need_shape_inference': True}).create_node_with_data()
            axis_const = Const(graph, {'value': int64_array(0), 'name': name + '/Axis'}).create_node_with_data()
            gather = Gather(graph, {'name': name,
                                    'need_shape_inference': True}).create_node_with_data([data_node, const, axis_const])
            attrs = graph.get_edge_data(data_node.id, node.id, key=0).copy()

            graph.add_edge(gather.id, node.id, **attrs)
            graph.remove_edge(data_node.id, node.id)
