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
import numpy as np

from extensions.back.FuseReshapesSequence import FuseReshapesSequence
from extensions.back.FuseTransposesSequence import FuseTransposesSequence
from extensions.back.TransposeToPermute import TransposeToPermute
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph


class ShuffleChannelPatternOptimization(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [FuseTransposesSequence, FuseReshapesSequence]

    def run_before(self):
        return [TransposeToPermute]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('t_start_order', {'type': 'Const'}),
                ('t_start_order_d', {'value': lambda value: value is not None and np.array_equal(value, [0, 2, 3, 1])}),
                ('t_start', {'type': 'Transpose'}),
                ('t_start_d', {}),

                ('reshape_dim', {'type': 'Const'}),
                ('reshape_dim_d', {'value': lambda value: value is not None and value.size == 5 and value[0] == -1}),
                ('reshape_start', {'type': 'Reshape'}),
                ('reshape_start_d', {}),

                ('t_5d_order', {'type': 'Const'}),
                ('t_5d_order_d', {'value': lambda value: value is not None and np.array_equal(value, [0, 1, 2, 4, 3])}),
                ('t_5d', {'type': 'Transpose'}),
                ('t_5d_d', {}),

                ('reshape_1_dim', {'type': 'Const'}),
                ('reshape_1_dim_d', {'value': lambda value: value is not None and value.size == 4 and value[0] == -1}),
                ('reshape_end', {'type': 'Reshape'}),
                ('reshape_end_d', {}),

                ('t_end_order', {'type': 'Const'}),
                ('t_end_order_d', {'value': lambda value: value is not None and np.array_equal(value, [0, 3, 1, 2])}),
                ('t_end', {'type': 'Transpose'}),
            ],
            edges=[
                ('t_start_order', 't_start_order_d'),
                ('t_start_order_d', 't_start', {'in': 1}),
                ('t_start', 't_start_d'),

                ('reshape_dim', 'reshape_dim_d'),
                ('t_start_d', 'reshape_start', {'in': 0}),
                ('reshape_dim_d', 'reshape_start', {'in': 1}),
                ('reshape_start', 'reshape_start_d'),

                ('t_5d_order', 't_5d_order_d'),
                ('reshape_start_d', 't_5d', {'in': 0}),
                ('t_5d_order_d', 't_5d', {'in': 1}),
                ('t_5d', 't_5d_d'),

                ('reshape_1_dim', 'reshape_1_dim_d'),
                ('t_5d_d', 'reshape_end', {'in': 0}),
                ('reshape_1_dim_d', 'reshape_end', {'in': 1}),
                ('reshape_end', 'reshape_end_d'),

                ('t_end_order', 't_end_order_d'),
                ('reshape_end_d', 't_end', {'in': 0}),
                ('t_end_order_d', 't_end', {'in': 1}),
            ],
        )

    @staticmethod
    def feature_dim_splitted(short_shape, long_shape):
        return all([short_shape[i] == long_shape[i] for i in range(len(short_shape) - 1)]) and \
               short_shape[-1] == long_shape[-1] * long_shape[-2]

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape_5d = match['reshape_start']
        if not ShuffleChannelPatternOptimization.feature_dim_splitted(
                short_shape=reshape_5d.in_port(0).data.get_shape(), long_shape=reshape_5d.out_port(0).data.get_shape()):
            return

        reshape_4d = match['reshape_end']
        if not ShuffleChannelPatternOptimization.feature_dim_splitted(
                short_shape=reshape_4d.out_port(0).data.get_shape(), long_shape=reshape_4d.in_port(0).data.get_shape()):
            return

        start = match['t_start']
        end = match['t_end']

        new_start = match['reshape_start']
        new_end = match['reshape_end']

        start_source = start.in_port(0).get_connection().get_source()
        end_connection = end.out_port(0).get_connection()

        new_end.out_port(0).disconnect()
        end_connection.set_source(new_end.out_port(0))

        start.in_port(0).disconnect()
        new_start.in_port(0).disconnect()

        new_start.in_port(0).connect(start_source)

        match['reshape_dim']['value'] = int64_array(np.take(new_start.in_port(1).data.get_value(), [0, 3, 4, 1, 2]))
        match['reshape_dim'].infer(match['reshape_dim'])
        new_start.infer(new_start)

        match['t_5d_order']['value'] = int64_array([0, 2, 1, 3, 4])
        match['t_5d_order'].infer(match['t_5d_order'])
        match['t_5d'].infer(match['t_5d'])

        match['reshape_1_dim']['value'] = int64_array(np.take(new_start.in_port(1).data.get_value(), [0, 3, 1, 2]))
        match['reshape_1_dim'].infer(match['reshape_1_dim'])
