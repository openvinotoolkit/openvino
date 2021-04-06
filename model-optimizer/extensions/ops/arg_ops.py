# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


def arg_ops_infer(node: Node):
    shape = node.in_node(0).shape
    if shape is None:
        return

    # there are two inputs in TensorFlow. The second input is the axis for ArgMax
    if len(node.in_nodes()) == 2:
        if node.in_node(1).value is None:
            log.debug('The second argument to {} is None'.format(node.soft_get('name', node.id)))
            return
        node.axis = node.in_node(1).value.item()
        # remove the unnecessary input
        node.graph.remove_edge(node.in_node(1).id, node.id)

    num_top_axes = shape.size
    if num_top_axes < 3:
        num_top_axes = 3

    out_shape = np.ones(num_top_axes, dtype=int)

    if node.has_valid('axis'):
        axis = get_canonical_axis_index(shape, node.axis)
        node.axis = axis
        out_shape = np.array(shape)
        out_shape[axis] = node.top_k
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
    else:
        out_shape[0] = shape[0]
        out_shape[2] = node.top_k
        if node.has_and_set('out_max_val'):
            out_shape[1] = 2

    node.out_node().shape = out_shape


class ArgMaxOp(Op):
    op = 'ArgMax'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': arg_ops_infer,
            'output_type': np.int64,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'out_max_val',
            'top_k',
            'axis',
        ]


class ArgMinOp(Op):
    op = 'ArgMin'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'infer': arg_ops_infer,
            'output_type': np.int64,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'top_k',
            'axis',
        ]