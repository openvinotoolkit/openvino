# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.pass_separator import PostMiddleStart, MiddleFinish
from openvino.tools.mo.middle.passes.eliminate import remove_op_node_with_data_node
from openvino.tools.mo.middle.passes.fusing.helpers import get_next_operation
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class FuseReshapesSequence(MiddleReplacementPattern):
    """
    Finds sequence of Reshapes operations and merge them to a single Reshape operation.
    """
    # TODO the pass should be extended for Reshape with special symbols "0" or "-1"
    # For example: 1,100 -> Reshape(2,5,10) -> 2,5,10 -> Reshape(0,10,-1) -> 2,10,5

    enabled = True
    run_not_recursively = True  # non-unified data nodes view in TI body (no Const ops, bare data node)

    def run_before(self):
        return [PostMiddleStart]

    def run_after(self):
        return [MiddleFinish]

    def find_and_replace_pattern(self, graph: Graph):
        reshape_nodes = graph.get_op_nodes(type='Reshape')
        for node in reshape_nodes:
            if not graph.has_node(node.id):
                # the Reshape node has been removed in the previous iteration
                continue
            if len(node.out_port(0).get_destinations()) == 1:
                log.debug('First phase for Reshape: {}'.format(node.soft_get('name')))

                next_op = get_next_operation(node)[0]
                log.debug('second node: id={}, type={}'.format(next_op.soft_get('id'), next_op.soft_get('type')))
                if next_op.has_valid('type') and next_op.type == 'Reshape':
                    dim_value = next_op.in_port(1).data.get_value()
                    if dim_value is None or 0 in dim_value or -1 in dim_value:
                        # we do not fuse reshape sequences with special symbols: 0, -1
                        continue

                    # Detected Reshape1 --> data --> Reshape2 pattern without side edges. Remove Reshape1
                    log.debug('Second phase for Reshape: {}'.format(node.soft_get('name')))
                    remove_op_node_with_data_node(graph, node)


class FuseReshapesSequenceKaldi(MiddleReplacementPattern):
    """
    Finds sequence of Reshapes operations of special type and remove them. It is enabled for Kaldi because
    such type of reshapes are created in add_reshape_around_convolution/pooling
    data(b, t, w, c) -> Reshape(0, -1) -> data(b, t*w*c) -> Reshape(br, tr, wr, cr)
    Check, that
    * br = b - taken from shape before Reshape as is;
    * tr = t and wr = w - that constants used in the second reshape is the same in shape before the first Reshape
    """

    enabled = True
    run_not_recursively = True  # non-unified data nodes view in TI body (no Const ops, bare data node)
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    def run_before(self):
        from openvino.tools.mo.middle.MergeNodesPermutations import MergeNodesPermutations
        return [MergeNodesPermutations]

    def run_after(self):
        return [FuseReshapesSequence]

    def pattern(self):
        return dict(
            nodes=[
                ('reshape_in_dims', dict(kind='op', op='Const')),
                ('reshape_in_dims_d', dict(kind='data')),
                ('reshape_in', dict(kind='op', op='Reshape', special_zero=True)),
                ('reshape_in_d', dict(kind='data')),
                ('shape', dict(kind='op', op='ShapeOf')),
                ('shape_d', dict(kind='data')),
                ('gather_in_1', dict(kind='op', op='Const')),
                ('gather_in_1_d', dict(kind='data')),
                ('gather_in_2', dict(kind='op', op='Const')),
                ('gather_in_2_d', dict(kind='data')),
                ('gather_batch', dict(kind='op', op='Gather')),
                ('gather_batch_d', dict(kind='data')),
                ('time_dim', dict(kind='op', op='Const')),
                ('time_dim_d', dict(kind='data')),
                ('concat_dims', dict(kind='op', op='Concat')),
                ('concat_dims_d', dict(kind='data')),
                ('reshape_out', dict(kind='op', op='Reshape')),
                ],
            edges=[('reshape_in_dims', 'reshape_in_dims_d'), ('reshape_in_dims_d', 'reshape_in', {'in': 1}),
                   ('reshape_in', 'reshape_in_d'), ('reshape_in_d', 'reshape_out', {'in': 0}),
                   ('reshape_in_d', 'shape'), ('shape', 'shape_d'),
                   ('shape_d', 'gather_batch', {'in': 0}),
                   ('gather_in_1', 'gather_in_1_d'), ('gather_in_1_d', 'gather_batch', {'in': 1}),
                   ('gather_in_2', 'gather_in_2_d'), ('gather_in_2_d', 'gather_batch', {'in': 2}),
                   ('gather_batch', 'gather_batch_d'), ('gather_batch_d', 'concat_dims', {'in': 0}),
                   ('time_dim', 'time_dim_d'), ('time_dim_d', 'concat_dims', {'in': 1}),
                   ('concat_dims', 'concat_dims_d'), ('concat_dims_d', 'reshape_out', {'in': 1})
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        reshape_in = match['reshape_in']

        log.debug('First phase for Reshape: {}'.format(reshape_in.soft_get('name')))
        in_shape = reshape_in.in_port(0).get_source().data.get_shape()
        if not is_fully_defined(in_shape[1:]):
            return

        reshape_in_dims = match['reshape_in_dims']
        if not np.all(reshape_in_dims.out_port(0).data.get_value() == [0, -1]):
            return

        gather_in_1 = match['gather_in_1']
        if not np.all(gather_in_1.out_port(0).data.get_value() == [0]):
            return

        gather_in_2 = match['gather_in_2']
        if not np.all(gather_in_2.out_port(0).data.get_value() == [0]):
            return

        reshape_out = match['reshape_out']
        log.debug('second child_node: id={}, type={}'.format(reshape_out.soft_get('id'), reshape_out.soft_get('type')))

        concat_dims_node = match['concat_dims']
        shapeof_node = match['shape']

        # check that t and w is the same as before the first Reshape
        t_node = match['time_dim']
        w_node = concat_dims_node.in_port(3).get_source().node
        const_dim_2 = 3
        if w_node.op != 'Const':
            w_node = concat_dims_node.in_port(2).get_source().node
            const_dim_2 = 2
        if w_node.op != 'Const' or \
                not is_fully_defined(t_node.out_port(0).data.get_value()) or \
                not is_fully_defined(w_node.out_port(0).data.get_value()) or \
                not np.all(t_node.out_port(0).data.get_value() == [in_shape[1]]) or \
                not np.all(w_node.out_port(0).data.get_value() == [in_shape[const_dim_2]]):
            return

        # Detected Reshape1 --> data --> Reshape2 pattern without side edges. Remove Reshape1
        log.debug('Second phase for Reshape: {}'.format(reshape_in.soft_get('name')))
        shapeof_node.in_port(0).disconnect()
        concat_dims_node.out_port(0).disconnect()
        remove_op_node_with_data_node(graph, reshape_in)
        remove_op_node_with_data_node(graph, reshape_out)
