# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.elementwise import Equal
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.select import Select
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes, Node
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class ExpandRangeConstant(FrontReplacementSubgraph):
    r"""
    Searches for Constant operations filled with range values starting from 0 and replaces it with Range operation
    Faced in ONNX BERT -- replacing it makes model reshape-able by sequence length

    WARNING: true BIDIRECTIONAL mode of Broadcast could cause issues
    (the probability is small, so we decided to keep the optimization)

    value_input[1, X] (value=range(0,X))     shape_input[Y, 1]
            \                               /
          Broadcast(mode='bidirectional') [Y, X]
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Broadcast'):
            value = node.in_port(0).get_source().node
            if value.soft_get('type') == 'Const':
                self.replace(node, value)

    @staticmethod
    def replace(node: Node, const: Node):
        graph = node.graph
        shape = const.shape
        const_name = const.soft_get('name', const.id)

        non_one_dims = np.argwhere(shape != 1).flatten()
        one_dims = np.argwhere(shape == 1).flatten()

        if not (non_one_dims.size == 1 and 5 < np.prod(shape) < 500):
            # (5;500) range is deduced to affect less models
            return

        value = const.value
        if not np.array_equal(np.arange(0, np.prod(shape), 1).reshape(shape), value):
            return

        positive_idx = non_one_dims.item(0)
        negative_idx = positive_idx - len(shape)

        node_name = node.soft_get('name', node.id)
        gather = create_op_with_const_inputs(graph, Gather, {1: int64_array(negative_idx), 2: int64_array(0)},
                                             {'name': node_name + '/BroadcastingDim'})
        gather_for_const = create_op_with_const_inputs(graph, Gather, {1: int64_array(negative_idx), 2: int64_array(0)},
                                                       {'name': const_name + '/BroadcastingDim'})
        shapeof_node = Shape(graph, {'name': const_name + '/ShapeOf'}).create_node()
        shapeof_node.out_port(0).connect(gather_for_const.in_port(0))

        equal_node = create_op_with_const_inputs(graph, Equal, {1: int64_array(1)}, {'name': node_name + '/ConstOne'})
        gather.out_port(0).connect(equal_node.in_port(0))

        select_node = Select(graph, {'name': node_name + '/Select',
                                      'auto_broadcast': 'numpy'}).create_node([equal_node, gather_for_const, gather])

        const.out_port(0).connect(shapeof_node.in_port(0))

        range_node = create_op_with_const_inputs(graph, Range,
                                                 {0: mo_array(0, dtype=value.dtype),
                                                  2: mo_array(1, dtype=value.dtype)},
                                                 {'name': const_name + '/Range', 'dtype': value.dtype})
        select_node.out_port(0).connect(range_node.in_port(1))

        node.in_port(1).get_connection().add_destination(gather.in_port(0))

        node.in_port(0).get_connection().set_source(range_node.out_port(0))

        if one_dims.size:
            unsqueeze = create_op_node_with_second_input(graph, Unsqueeze, one_dims,
                                                         {'name': const_name + '/KeepShape'})
            range_node.out_port(0).get_connection().insert_node(unsqueeze)
            rename_nodes([(const, const_name + '/ToBeDeleted'), (unsqueeze, const_name)])
        else:
            rename_nodes([(const, const_name + '/ToBeDeleted'), (range_node, const_name)])
