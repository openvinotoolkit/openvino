# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


class DivisionToZeroFP16ResolverMaximumEps(MiddleReplacementPattern):
    """
    """
    enabled = True
    # graph_condition = [lambda graph: graph.graph['cmd_params'].data_type == 'FP16']

    def run_after(self):
        from extensions.middle.fusings import Fusing
        return [Fusing]

    def run_before(self):
        from extensions.middle.L2NormFusing import L2NormToNorm
        return [L2NormToNorm]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('eps_or_input_data_1', dict(kind='data')),  # one of these inputs is eps
                ('eps_or_input_data_2', dict(kind='data')),
                ('max', dict(kind='op', op='Maximum')),
                ('max_data', dict(kind='data')),
                ('pow_exp', dict(kind='data', value=lambda x: np.all(x < -0) if x is not None else False)),
                ('pow', dict(kind='op', op='Pow')),
                ('pow_data', dict(kind='data')),
                ('multiplicative_inverse', dict(kind='op', op='Mul')),
            ],
            edges=[
                ('eps_or_input_data_1', 'max'),
                ('eps_or_input_data_2', 'max'),
                ('max', 'max_data'),
                ('max_data', 'pow', {'in': 0}),
                ('pow_exp', 'pow', {'in': 1}),
                ('pow', 'pow_data'),
                ('pow_data', 'multiplicative_inverse'),
                ('input', 'multiplicative_inverse'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        change_const_value(match['max'])


class DivisionToZeroFP16ResolverAddEpsilon(MiddleReplacementPattern):
    """
    """
    enabled = True
    # graph_condition = [lambda graph: graph.graph['cmd_params'].data_type == 'FP16']

    def run_after(self):
        from extensions.middle.fusings import Fusing
        return [Fusing]

    def run_before(self):
        from extensions.middle.L2NormFusing import L2NormToNorm
        return [L2NormToNorm]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('eps_or_input_data_1', dict(kind='data')),  # one of these inputs is eps
                ('eps_or_input_data_2', dict(kind='data')),
                ('add', dict(kind='op', op='Add')),
                ('add_data', dict(kind='data')),
                ('pow_exp', dict(kind='data', value=lambda x: np.all(x < -0) if x is not None else False)),
                ('pow', dict(kind='op', op='Pow')),
                ('pow_data', dict(kind='data')),
                ('multiplicative_inverse', dict(kind='op', op='Mul')),
            ],
            edges=[
                ('eps_or_input_data_1', 'add', {'in': 0}),
                ('eps_or_input_data_2', 'add', {'in': 1}),
                ('add', 'add_data'),
                ('add_data', 'pow', {'in': 0}),
                ('pow_exp', 'pow', {'in': 1}),
                ('pow', 'pow_data'),
                ('pow_data', 'multiplicative_inverse'),
                ('input', 'multiplicative_inverse'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        change_const_value(match['add'])


def change_const_value(node: Node):  # node is either max or  add
    is_port_1_const = node.in_port(1).get_source().node.soft_get('op') == 'Const'
    port = 1 if is_port_1_const else 0
    const_node = node.in_port(port).get_source().node
    const_name = const_node.soft_get('name', const_node.id)

    fp16_machine_eps = np.finfo(np.float16).eps
    if const_node.value is not None and const_node.value < fp16_machine_eps:
        fp16_machine_eps = np.array(fp16_machine_eps, dtype=const_node.value.dtype)
        log.error('changing value of constant {} from {} to {} to '
                  'prevent division to zero'.format(const_name, const_node.value, fp16_machine_eps),
                  extra={'is_warning': True})
        const_node.value = fp16_machine_eps
        const_node.out_port(0).data.set_value(fp16_machine_eps)
