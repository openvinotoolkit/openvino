"""
 Copyright (C) 2018-2020 Intel Corporation

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
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_node
from mo.middle.replacement import MiddleReplacementPattern


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

        # rename l2_normalize node since it will be no longer output after the transformation
        output_name = match['l2_normalize'].soft_get('name', match['l2_normalize'].id)
        normalizel2_name = output_name + '/normalizel2'
        rename_node(match['l2_normalize'], normalizel2_name)

        normalize_node = create_op_node_with_second_input(graph, NormalizeOp,
                                                          np.ones(shape=int64_array([match['input'].shape[-1]]),
                                                                  dtype=match['input'].data_type),
                                                          {'name': output_name, 'eps': y,
                                                           'across_spatial': 0, 'channel_shared': 0})
        rename_node(normalize_node, output_name)

        match['square'].in_port(0).get_source().connect(normalize_node.in_port(0))

        match['square'].in_port(0).disconnect()
        if match['l2_normalize'].in_port(0).get_source().node.id == match['rsqrt'].id:
            match['l2_normalize'].in_port(1).disconnect()
        else:
            match['l2_normalize'].in_port(0).disconnect()

        match['l2_normalize'].out_port(0).get_connection().set_source(normalize_node.out_port(0))
