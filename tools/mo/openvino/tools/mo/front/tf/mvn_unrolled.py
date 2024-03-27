# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.PowerToEltwises import PowerToEltwises
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Node, Graph


class MVNUnrolled(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        return [PowerToEltwises]

    def pattern(self):
        log.debug('Enabled MVN replacement')
        return dict(
            nodes=[
                ('mean', dict(kind='op', op='ReduceMean')),
                ('stop_grad', dict(kind='op', op='StopGradient')),
                ('sqdiff', dict(kind='op', op='SquaredDifference')),
                ('variance', dict(kind='op', op='ReduceMean')),
                ('add', dict(kind='op', op='Add')),
                ('pow', dict(kind='op', op='Pow')),
                ('sub', dict(kind='op', op='Sub')),
                ('truediv', dict(kind='op', op='Div')),
            ],
            edges=[
                ('mean', 'stop_grad', {'in': 0}),
                ('stop_grad', 'sqdiff', {'in': 1}),
                ('sqdiff', 'variance', {'in': 0}),
                ('mean', 'sub', {'in': 1}),
                ('variance', 'add'),
                ('add', 'pow', {'in': 0}),
                ('pow', 'truediv', {'in': 1}),
                ('sub', 'truediv', {'in': 0}),
            ])

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        mvn = MVN(graph, dict(
            name=match['truediv'].name + '/MVN_',
            eps_mode='outside_sqrt',
            normalize_variance=1
        ))
        mvn.attrs['old_infer'] = mvn.attrs['infer']
        mvn.attrs['infer'] = __class__.infer

        mean_reduction = match['mean'].in_node(1)
        variance_reduction = match['variance'].in_node(1)
        pow2 = match['pow'].in_node(1)
        eps = match['add'].in_node(0 if match['add'].in_node(0).id != match['variance'].id else 1)

        new_subgraph = mvn.create_node([match['mean'].in_node(0), mean_reduction, variance_reduction, pow2, eps])

        match['truediv'].replace_node(new_subgraph)

    @staticmethod
    def infer(node: Node):
        axes_1_value = node.in_port(1).data.get_value()
        axes_2_value = node.in_port(2).data.get_value()
        if axes_1_value is None or axes_2_value is None:
            log.warning('Reduction indices for mean and variance for MVN node {} are not constants'.format(node.name))
            return

        if not (all(axes_1_value == axes_2_value)):
            log.warning('Reduction indices for mean {} and variance {} do not match'.format(
                axes_1_value,
                axes_2_value
            ))
            return

        power_value = node.in_port(3).data.get_value()
        eps_value = node.in_port(4).data.get_value()
        if power_value is None or eps_value is None:
            log.warning('Power or/and epsilon values for MVN node {} are not constants'.format(node.name))
            return

        if power_value != 0.5:
            log.warning('Power for MVN node {} ({}) is not equal to 0.5'.format(node.name, power_value))
            return

        node['eps'] = eps_value

        for i in range(2, 5):
            node.in_port(i).disconnect()
        node.old_infer(node)
        node.infer = node.old_infer
        del node['old_infer']
