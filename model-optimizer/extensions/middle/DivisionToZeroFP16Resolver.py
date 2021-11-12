# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


class DivisionToZeroFP16Resolver(MiddleReplacementPattern):
    """
    Patterns input_1/Maximum(input_2, eps) and input_1/Add(input_2, eps) are used
    to prevent division to zero. But usually in FP32 networks eps is such
    small (e.g. 1e-9, 1e-12, ...) that after casting to FP16 it's collapsed to zero.
    This can lead to division to zero if input_2 is also zero.
    To prevent that we change eps to FP16 smallest normal value in such patterns.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].compress_fp16]

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
                ('max_or_add', dict(kind='op', op=lambda x: x in ['Maximum', 'Add'])),
                ('max_or_add_data', dict(kind='data')),
                ('pow_exp', dict(kind='data', value=lambda x: np.all(x < 0) if x is not None else False)),
                ('pow', dict(kind='op', op='Pow')),
                ('pow_data', dict(kind='data')),
                ('multiplicative_inverse', dict(kind='op', op='Mul')),
            ],
            edges=[
                ('eps_or_input_data_1', 'max_or_add'),
                ('eps_or_input_data_2', 'max_or_add'),
                ('max_or_add', 'max_or_add_data'),
                ('max_or_add_data', 'pow', {'in': 0}),
                ('pow_exp', 'pow', {'in': 1}),
                ('pow', 'pow_data'),
                ('pow_data', 'multiplicative_inverse'),
                ('input', 'multiplicative_inverse'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        is_port_1_const = match['max_or_add'].in_port(1).get_source().node.soft_get('op') == 'Const'
        port = 1 if is_port_1_const else 0

        const_node = match['max_or_add'].in_port(port).get_source().node
        value = const_node.value
        # we use FP16 smallest normal value, because arithmetic of subnormal values is slower
        fp16_smallest_positive = np.finfo(np.float16).tiny

        if value is not None and np.all(value < fp16_smallest_positive):
            new_eps = np.full_like(value, fp16_smallest_positive)
            const_node.out_port(0).data.set_value(new_eps)

            const_name = const_node.soft_get('name', const_node.id)
            log.error("Changing value of constant '{}' from {} -> {} to "
                      "prevent division to zero when casted to FP16".format(const_name, value, new_eps),
                      extra={'is_warning': True})
