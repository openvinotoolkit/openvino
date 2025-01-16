# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.upsample import UpsampleOp
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph, Node


class BatchToSpaceToUpsample(FrontReplacementSubgraph):
    """
    The transformation looks for pattern that performs NX upscale of the input image specified in the NHWC layout.
    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.tf.space_to_batch import BatchToSpaceNormalizer
        return [BatchToSpaceNormalizer]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[
                ('transpose', dict(op='Transpose')),
                ('expand_dims', dict(op='Unsqueeze')),
                ('tile', dict(op='Tile')),
                ('batch_to_space_nd', dict(op='BatchToSpace')),
                ('strided_slice', dict(op='StridedSlice')),
                ('transpose_back', dict(op='Transpose')),
            ],
            edges=[
                ('transpose', 'expand_dims', {'out': 0}),
                ('expand_dims', 'tile', {'out': 0}),
                ('tile', 'batch_to_space_nd', {'out': 0}),
                ('batch_to_space_nd', 'strided_slice', {'out': 0}),
                ('strided_slice', 'transpose_back', {'out': 0})
            ]
        )

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict, **kwargs):
        def _input_node_value(node: Node, port_ind: int):
            input_node = node.in_port(port_ind).get_source().node
            return input_node.value if input_node.op == 'Const' else None

        transpose = match['transpose']
        transpose_order = _input_node_value(transpose, 1)
        if transpose_order is None or not np.all(np.equal(transpose_order, int64_array([1, 2, 3, 0]))):
            log.debug('The transpose order {} for node {} is not equal to [1, 2, 3, 0]. Cannot apply '
                      'BatchToSpaceToUpsample transformation.'.format(transpose_order, transpose.name))
            return

        expand_axis = match['expand_dims']
        expand_axis_value = _input_node_value(expand_axis, 1)
        if expand_axis_value != 0:
            log.debug('The expand axis {} for node {} is not equal to 0. Cannot apply BatchToSpaceToUpsample '
                      'transformation.'.format(expand_axis_value, expand_axis.name))
            return

        tile = match['tile']
        tile_value = _input_node_value(tile, 1)
        if tile_value is None:
            log.debug('The tile value is not defined for node {}. Cannot apply BatchToSpaceToUpsample '
                      'transformation.'.format(tile.name))
            return

        if len(np.where(tile_value != 1)) != 1:
            log.debug('The number of tiles not equal to 1 not equal to 1. Cannot apply BatchToSpaceToUpsample '
                      'transformation.')
            return
        tile_batch = tile_value[0]

        batch_to_space_nd = match['batch_to_space_nd']
        block_shape = _input_node_value(batch_to_space_nd, 1)
        if block_shape is None or tile_batch != np.prod(block_shape):
            log.debug('The block shape {} for node {} is not defined or inconsistent with the tile size. Cannot apply '
                      'BatchToSpaceToUpsample transformation.'.format(block_shape, batch_to_space_nd.name))
            return
        if len(block_shape) != 2:
            log.debug('The block shape len is not equal to 2 for node {}. Cannot apply BatchToSpaceToUpsample '
                      'transformation.'.format(batch_to_space_nd.name))
            return

        crops = _input_node_value(batch_to_space_nd, 2)
        if crops is None or np.count_nonzero(crops) != 0:
            log.debug('Crops for node {} are non zero. Cannot apply BatchToSpaceToUpsample '
                      'transformation.'.format(batch_to_space_nd.name))
            return

        transpose_back = match['transpose_back']
        transpose_back_order = _input_node_value(transpose_back, 1)
        if transpose_back_order is None or not np.all(np.equal(transpose_back_order, int64_array([3, 0, 1, 2]))):
            log.debug('The transpose order {} for node {} is not equal to [3, 0, 1, 2]. Cannot apply '
                      'BatchToSpaceToUpsample transformation.'.format(transpose_back_order, transpose_back.name))
            return

        upsample_node = UpsampleOp(graph, {'height_scale': block_shape[0], 'width_scale': block_shape[1],
                                           'mode': 'nearest',
                                           'name': transpose.name + '/upsample'}).create_node()

        match['transpose'].in_port(0).get_connection().set_destination(upsample_node.in_port(0))
        match['transpose_back'].out_port(0).get_connection().set_source(upsample_node.out_port(0))
