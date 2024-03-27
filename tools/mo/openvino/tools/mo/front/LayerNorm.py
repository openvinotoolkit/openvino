# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.pattern_match import apply_pattern


class LayerNorm(FrontReplacementPattern):
    # Compose part of the LayerNorm pattern to the MVN
    enabled = True

    def pattern1(self):
        return dict(
            nodes=[
                ('pool0', dict(op='ReduceMean')),
                ('pool1', dict(op='ReduceMean')),
                ('pow', dict(op='Pow')),
                ('div', dict(op='Div')),
                ('sqrt', dict(op='Pow')),
                ('add', dict(op='Add')),
                ('sub', dict(op='Sub')),
                ('pool0_param', dict(op='Const')),
                ('pool1_param', dict(op='Const')),
                ('add_param', dict(op='Const')),
                ('pow_param', dict(op='Const')),
            ],
            edges=[
                ('pool0', 'sub'),
                ('sub', 'pow'),
                ('pow', 'pool1'),
                ('pool1', 'add'),
                ('add', 'sqrt'),
                ('sqrt', 'div'),
                ('sub', 'div'),
                ('pool0_param', 'pool0'),
                ('pool1_param', 'pool1'),
                ('pow_param', 'sqrt'),
                ('add_param', 'add'),
            ])

    def pattern2(self):
        # pattern from bert onnx model
        return dict(
            nodes=[
                ('pool0', dict(op='ReduceMean')),
                ('pool1', dict(op='ReduceMean')),
                ('cast', dict(op='Cast')),
                ('pow', dict(op='Pow')),
                ('div', dict(op='Div')),
                ('sqrt', dict(op='Pow')),
                ('add', dict(op='Add')),
                ('sub', dict(op='Sub')),
                ('pool0_param', dict(op='Const')),
                ('pool1_param', dict(op='Const')),
                ('add_param', dict(op='Const')),
                ('pow_param', dict(op='Const')),
            ],
            edges=[
                ('pool0', 'sub'),
                ('sub', 'cast'),
                ('cast', 'pow'),
                ('pow', 'pool1'),
                ('pool1', 'add'),
                ('add', 'sqrt'),
                ('sqrt', 'div'),
                ('sub', 'div'),
                ('pool0_param', 'pool0'),
                ('pool1_param', 'pool1'),
                ('pow_param', 'sqrt'),
                ('add_param', 'add'),
            ])

    def find_and_replace_pattern(self, graph: Graph):
        log.info('Enabled LayerNorm pattern recognition')
        apply_pattern(graph, **self.pattern1(), action=self.replace_layer_norm)
        apply_pattern(graph, **self.pattern2(), action=self.replace_layer_norm)

    def replace_layer_norm(self, graph: Graph, match: dict):
        inp = match['pool0']
        node_before = inp.in_port(0).get_source().node
        node_before_name = node_before.soft_get('name', node_before.id)

        # take/check the values of the add, pow and axes for ReduceMean
        pow_param = match['pow_param']
        add_param = match['add_param']
        if add_param.value.size == 1 and pow_param.value.size == 1 and add_param.value.item() <= 1e-05 \
                and pow_param.value.item() == 0.5 and match['pool0_param'].value == match['pool1_param'].value:
            log.debug('Found LayerNorm pattern after {} with name {}'.format(node_before.op, node_before_name))
            mvn = create_op_with_const_inputs(graph, MVN, {1: match['pool1_param'].value},
                                              {'eps': add_param.value.item(), 'normalize_variance': 1,
                                               'eps_mode': 'inside_sqrt'})
            div_name = match['div'].soft_get('name', match['div'].id)
            rename_nodes([(match['div'], div_name + '/to_be_removed'), (mvn, div_name)])

            inp.in_port(0).get_connection().set_destination(mvn.in_port(0))
            match['div'].out_port(0).get_connection().set_source(mvn.out_port(0))
