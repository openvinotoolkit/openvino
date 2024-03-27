# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from openvino.tools.mo.middle.fusings import Fusing
from openvino.tools.mo.middle.pass_separator import PostMiddleStart
from openvino.tools.mo.ops.activation_ops import LeakyReLU
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class LeakyReLUFusion(MiddleReplacementPattern):
    """
    The transformation  finds next subgraph:

    -->Data-------->Maximum-->Data
          `-->Mul---`

    and replaces with ReLU with negative slope (LeakyRelu)
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [Fusing]

    def run_before(self):
        return [PostMiddleStart]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('data', dict(kind='data')),
                ('mul_data', dict(kind='data')),
                ('max_op', dict(kind='op', type='Maximum')),
                ('const_op', dict(kind='op', type='Const')),
                ('const_data', dict(kind='data')),
                ('mul_op', dict(kind='op', type='Multiply')),
            ],
            edges=[
                ('data', 'mul_op'),
                ('mul_op', 'mul_data'),
                ('data', 'max_op'),
                ('mul_data', 'max_op'),
                ('const_op', 'const_data'),
                ('const_data', 'mul_op')
            ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        mul_node = match['mul_op']
        const_node = match['const_op']
        max_node = match['max_op']
        max_name = max_node.soft_get('name', max_node.id)

        const_value = const_node.out_port(0).data.get_value()
        if const_value is None or const_value.size != 1:
            log.debug('Mul layer "{}" can not participate in conversion to the LeakyReLU because constant "{}" '
                      'contains more than one element: {}'.format(mul_node.id, const_node.id, const_value.size))
            return

        # Create new LeakyReLU operation
        leaky_relu_node = LeakyReLU(graph, dict(negative_slope=const_value.item(0))).create_node()

        data_in_port = int(mul_node.in_port(0).get_source().node.type == 'Const')
        mul_node.in_port(data_in_port).get_source().connect(leaky_relu_node.in_port(0))
        max_node.out_port(0).get_connection().set_source(leaky_relu_node.out_port(0))

        rename_nodes([(max_node, max_name + '/TBR'), (leaky_relu_node, max_name)])

        log.debug('Successful conversion from {} {} to ReLU with negative slope (leaky ReLU)'
                  ''.format(max_node.id, mul_node.id))
