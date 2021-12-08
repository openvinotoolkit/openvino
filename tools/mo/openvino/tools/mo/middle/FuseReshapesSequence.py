# Copyright (C) 2018-2021 Intel Corporation
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
    Finds sequence of Reshapes operations of sepcial type and remove them. It is enabled for Kaldi because
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

    def find_and_replace_pattern(self, graph: Graph):
        reshape_nodes = graph.get_op_nodes(type='Reshape')
        for node in reshape_nodes:
            if not graph.has_node(node.id):
                # the Reshape node has been removed in the previous iteration
                continue
            if len(node.out_port(0).get_destinations()) != 2:
                continue

            log.debug('First phase for Reshape: {}'.format(node.soft_get('name')))
            if node.in_port(1).get_source().node.op != 'Const' or \
               not np.all(node.in_port(1).get_source().data.get_value() == [0, -1]):
                continue

            in_shape = node.in_port(0).get_source().data.get_shape()
            if not is_fully_defined(in_shape[1:]):
                continue

            next_ops = node.out_port(0).get_destinations()
            next_op = next_ops[0].node
            if not next_op.has_valid('type') or next_op.type != 'Reshape':
                next_op = next_ops[1].node
                if not next_op.has_valid('type') or next_op.type != 'Reshape':
                    continue
            log.debug('second node: id={}, type={}'.format(next_op.soft_get('id'), next_op.soft_get('type')))

            shape_node = next_op.in_port(1).get_source().node
            if shape_node.op != 'Concat' or len(shape_node.in_nodes()) != 4:
                continue
            # check that batch is the same as in previous node
            gather_batch = shape_node.in_port(0).get_source().node
            if gather_batch.op != 'Gather' or \
                    not np.all(gather_batch.in_port(1).get_source().data.get_value() == [0]) or \
                    not np.all(gather_batch.in_port(2).get_source().data.get_value() == [0]):
                continue
            shapeof_node = gather_batch.in_port(0).get_source().node
            if shapeof_node.op != 'ShapeOf' or \
                    shapeof_node.in_port(0).get_source() != node.out_port(0):
                continue
            # check that t and w is the same as before the first Reshape
            t_node = shape_node.in_port(1).get_source().node
            w_node = shape_node.in_port(3).get_source().node
            const_dim_2 = 3
            if w_node.op != 'Const':
                w_node = shape_node.in_port(2).get_source().node
                const_dim_2 = 2
            if t_node.op != 'Const' or w_node.op != 'Const' or \
                    not is_fully_defined(t_node.out_port(0).data.get_value()) or \
                    not is_fully_defined(w_node.out_port(0).data.get_value()) or \
                    not np.all(t_node.out_port(0).data.get_value() == [in_shape[1]]) or \
                    not np.all(w_node.out_port(0).data.get_value() == [in_shape[const_dim_2]]):
                continue

            # Detected Reshape1 --> data --> Reshape2 pattern without side edges. Remove Reshape1
            log.debug('Second phase for Reshape: {}'.format(node.soft_get('name')))
            shapeof_node.in_port(0).disconnect()
            shape_node.out_port(0).disconnect()
            remove_op_node_with_data_node(graph, node)
            remove_op_node_with_data_node(graph, next_op)
