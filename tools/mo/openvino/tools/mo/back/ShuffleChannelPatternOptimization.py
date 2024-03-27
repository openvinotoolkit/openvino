# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.FuseTransposesSequence import FuseTransposesSequence
from openvino.tools.mo.ops.depth_to_space import DepthToSpaceOp
from openvino.tools.mo.ops.shufflechannel import ShuffleChannels
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph


class ShuffleChannelPatternOptimization(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [FuseTransposesSequence]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('t_start_order', {'type': 'Const'}),
                ('t_start_order_d',
                 {'value': lambda v: v is not None and np.all(np.array_equal(v, [0, 2, 3, 1]))}),
                ('t_start', {'type': 'Transpose'}),
                ('t_start_d', {}),

                ('reshape_dim', {'type': 'Const'}),
                ('reshape_dim_d',
                 {'value': lambda v: v is not None and v.size == 5 and np.all(v[0] == -1)}),
                ('reshape_start', {'type': 'Reshape'}),
                ('reshape_start_d', {}),

                ('t_5d_order', {'type': 'Const'}),
                ('t_5d_order_d', {'value': lambda v: v is not None and np.all(np.array_equal(v, [0, 1, 2, 4, 3]))}),
                ('t_5d', {'type': 'Transpose'}),
                ('t_5d_d', {}),

                ('reshape_1_dim', {'type': 'Const'}),
                ('reshape_1_dim_d', {'value': lambda v: v is not None and v.size == 4 and np.all(v[0] == -1)}),
                ('reshape_end', {'type': 'Reshape'}),
                ('reshape_end_d', {}),

                ('t_end_order', {'type': 'Const'}),
                ('t_end_order_d', {'value': lambda v: v is not None and np.all(np.array_equal(v, [0, 3, 1, 2]))}),
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

        match['reshape_1_dim']['value'] = int64_array(np.take(new_end.in_port(1).data.get_value(), [0, 3, 1, 2]))
        match['reshape_1_dim'].infer(match['reshape_1_dim'])


class ShuffleChannelFusion(BackReplacementPattern):
    """
    FUSION: Reshape->Transpose->Reshape  to  ShuffleChannel
    We are able to perform the fusion if the pattern satisfies the conditions:
    1. Pattern input 4D shape is the same as pattern output 4D shape
    2. First Reshape splits channel dimension (1 axis) into two dimensions
    3. Transpose permutes only split dimensions
    4. Second Reshape pack them back

    Fixes original models reshape-ability (Smart reshape)
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [FuseTransposesSequence]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('reshape_0_pattern', dict(type='Const')),
                ('reshape_0_pattern_d', dict(value=lambda v: v is not None and v.size == 5 and np.all(v > 0))),
                ('reshape_0', dict(type='Reshape')),
                ('reshape_0_d', dict()),

                ('order', dict(type='Const')),
                ('order_d', dict(value=lambda v: v is not None and np.array_equal([0, 2, 1, 3, 4], v))),
                ('transpose', dict(type='Transpose')),
                ('transpose_d', {}),

                ('reshape_1_pattern', dict(type='Const')),
                ('reshape_1_pattern_d', dict(value=lambda v: v is not None and v.size == 4 and np.all(v > 0))),
                ('reshape_1', dict(type='Reshape')),
            ],
            edges=[
                ('reshape_0_pattern', 'reshape_0_pattern_d'),
                ('reshape_0_pattern_d', 'reshape_0'),
                ('reshape_0', 'reshape_0_d'),
                ('reshape_0_d', 'transpose'),
                ('order', 'order_d'),
                ('order_d', 'transpose'),
                ('transpose', 'transpose_d'),
                ('transpose_d', 'reshape_1'),
                ('reshape_1_pattern', 'reshape_1_pattern_d'),
                ('reshape_1_pattern_d', 'reshape_1'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        channel_splitting_reshape = match['reshape_0']
        channel_concating_reshape = match['reshape_1']

        initial_shape = channel_splitting_reshape.in_port(0).data.get_shape()
        resulting_shape = channel_concating_reshape.in_port(1).data.get_value()
        if not np.array_equal(initial_shape, resulting_shape):
            return

        channel_splitted_out_shape = channel_splitting_reshape.in_port(1).data.get_value()
        if not all([initial_shape[i] == channel_splitted_out_shape[j] for i, j in {0: 0, 2: 3, 3: 4}.items()]):
            return

        name = channel_concating_reshape.soft_get('name', channel_concating_reshape.id)
        group = channel_splitted_out_shape[1]
        shuffle_channel = ShuffleChannels(graph, {'name': name, 'group': group}).create_node()
        channel_concating_reshape.out_port(0).get_connection().set_source(shuffle_channel.out_port(0))
        shuffle_channel.in_port(0).connect(channel_splitting_reshape.in_port(0).get_source())


class DepthToSpaceFusion(BackReplacementPattern):
    """
    FUSION: Reshape->Transpose->Reshape  to  DepthToSpace
    We are able to perform the fusion if the pattern satisfies the conditions:
    1. Pattern has 6D input and 4D output
    2. First Reshape splits channel dimension (1 axis) into three dimensions [new_depth, block_size, block_size]
    3. Transpose permutes split dimensions with spatial ones
    4. Second Reshape pack block size together with spatial dimension

    Fixes original models reshape-ability (Smart reshape)
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [FuseTransposesSequence]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('reshape_0_pattern', dict(type='Const')),
                ('reshape_0_pattern_d', dict(value=lambda v: v is not None and v.size == 6 and np.all(v > 0))),
                ('reshape_0', dict(type='Reshape')),
                ('reshape_0_d', dict()),

                ('order', dict(type='Const')),
                ('order_d', dict(value=lambda v: v is not None and np.array_equal([0, 1, 4, 2, 5, 3], v))),
                ('transpose', dict(type='Transpose')),
                ('transpose_d', {}),

                ('reshape_1_pattern', dict(type='Const')),
                ('reshape_1_pattern_d', dict(value=lambda v: v is not None and v.size == 4 and np.all(v > 0))),
                ('reshape_1', dict(type='Reshape')),
            ],
            edges=[
                ('reshape_0_pattern', 'reshape_0_pattern_d'),
                ('reshape_0_pattern_d', 'reshape_0'),
                ('reshape_0', 'reshape_0_d'),
                ('reshape_0_d', 'transpose'),
                ('order', 'order_d'),
                ('order_d', 'transpose'),
                ('transpose', 'transpose_d'),
                ('transpose_d', 'reshape_1'),
                ('reshape_1_pattern', 'reshape_1_pattern_d'),
                ('reshape_1_pattern_d', 'reshape_1'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        channel_splitting_reshape = match['reshape_0']
        channel_concating_reshape = match['reshape_1']

        initial_shape = channel_splitting_reshape.in_port(0).data.get_shape()
        resulting_shape = channel_concating_reshape.in_port(1).data.get_value()
        if initial_shape[0] != resulting_shape[0]:
            return

        channel_splitted_out_shape = channel_splitting_reshape.in_port(1).data.get_value()
        if not all([initial_shape[i] == channel_splitted_out_shape[j] for i, j in {0: 0, 2: 4, 3: 5}.items()]) or \
                channel_splitted_out_shape[1] != channel_splitted_out_shape[2]:
            return
        block_size = channel_splitted_out_shape[2]
        expected_output_shape = [initial_shape[0], initial_shape[1] // (block_size * block_size),
                                 initial_shape[2] * block_size, initial_shape[3] * block_size]
        if not np.array_equal(expected_output_shape, resulting_shape):
            return

        name = channel_concating_reshape.soft_get('name', channel_concating_reshape.id)
        depth_to_space = DepthToSpaceOp(graph,
                                        {'name': name, 'block_size': block_size, 'mode': 'depth_first'}).create_node()
        channel_concating_reshape.out_port(0).get_connection().set_source(depth_to_space.out_port(0))
        depth_to_space.in_port(0).connect(channel_splitting_reshape.in_port(0).get_source())
