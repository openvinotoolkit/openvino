# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


def arg_ops_infer(node: Node):
    shape = node.in_port(0).data.get_shape()
    node_name = node.soft_get('name', node.id)
    assert shape is not None, "Input shape for the node {} is None".format(node_name)

    # there are two inputs in TensorFlow. The second input is the axis for ArgMax
    connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
    if len(connected_in_ports) == 2:
        axis = node.in_port(1).data.get_value()
        if axis is None:
            log.debug('The second argument to {} is None'.format(node.soft_get('name', node.id)))
            return
        node.axis = axis
        # remove the unnecessary input
        node.in_port(1).disconnect()

    num_top_axes = shape.size
    if num_top_axes < 3:
        num_top_axes = 3

    out_shape = np.ones(num_top_axes, dtype=np.int64)

    if node.has_valid('axis'):
        axis = get_canonical_axis_index(shape, node.axis)
        node.axis = axis
        out_shape = int64_array(shape)
        out_shape[axis] = node.top_k
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
    else:
        out_shape[0] = shape[0]
        out_shape[2] = node.top_k
        if node.has_and_set('out_max_val'):
            out_shape[1] = 2

    node.out_port(0).data.set_shape(out_shape)


class ArgMaxOp(Op):
    op = 'ArgMax'
    enabled = False

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
