"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log

import numpy as np

from extensions.ops.normalize import NormalizeOp
from mo.front.common.layout import get_features_dim
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class L2NormToNorm(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('l2_normalize', dict(kind='op', op='Mul')),
                ('l2_normalize_data', dict(kind='data')),
                ('maximum', dict(kind='op', op='Maximum')),
                ('maximum_data', dict(kind='data')),
                ('maximum_y_data', dict(kind='data')),
                ('rsqrt_pow', dict(kind='data', value=lambda x: np.all(x == -0.5) if x is not None else False)),
                ('rsqrt', dict(kind='op', op='Pow')),
                ('rsqrt_data', dict(kind='data')),
                ('square_pow', dict(kind='data', value=lambda x: np.all(x == 2) if x is not None else False)),
                ('square', dict(kind='op', op='Pow')),
                ('square_data', dict(kind='data')),
                ('sum', dict(kind='op', op='ReduceSum')),
                ('sum_data', dict(kind='data')),
            ],
            edges=[
                ('input', 'square', {'in': 0}),
                ('square_pow', 'square', {'in': 1}),
                ('square', 'square_data'),
                ('square_data', 'sum'),
                ('sum', 'sum_data'),
                ('maximum_y_data', 'maximum'),
                ('sum_data', 'maximum'),
                ('maximum', 'maximum_data'),
                ('maximum_data', 'rsqrt', {'in': 0}),
                ('rsqrt_pow', 'rsqrt', {'in': 1}),
                ('rsqrt', 'rsqrt_data'),
                ('rsqrt_data', 'l2_normalize'),
                ('input', 'l2_normalize'),
                ('l2_normalize', 'l2_normalize_data'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        y = match['maximum'].in_port(0).data.get_value()
        if y is None:
            y = match['maximum'].in_port(1).data.get_value()

        if y is None or y.shape != ():
            log.debug('The value of the "maximum_y_data" is not defined or is not constant')
            return

        normalize_input_node = match['square'].in_port(0).get_source().node
        normalize_node = NormalizeOp(graph, {'name': normalize_input_node.soft_get('name') + '/Normalize', 'eps': y,
                                             'across_spatial': 0, 'channel_shared': 0}).create_node()

        weights_node = Const(graph, {'value': np.ones(shape=int64_array([match['input'].shape[-1]]),
                                                      dtype=match['input'].data_type)}).create_node()

        # the normalize_input_node has 2 consumers so it is necessary to disconnect output port first
        normalize_input_node.out_port(0).disconnect()
        normalize_input_node.out_port(0).get_connection().set_destination(normalize_node.in_port(0))
        weights_node.out_port(0).get_connection().set_destination(normalize_node.in_port(1))
        match['l2_normalize'].out_port(0).get_connection().set_source(normalize_node.out_port(0))
