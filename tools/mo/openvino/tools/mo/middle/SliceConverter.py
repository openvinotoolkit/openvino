# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.clamp import Clamp
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.strided_slice import StridedSlice


def create_ss_interval_border(graph: Graph, slice_border_port: Port, shape: np.ndarray, axes: np.ndarray, node_name: str):
    """
    This function creates "begin"/"end" parameters for the StridedSlice based on Slice's "starts"/"ends"

    :param graph: graph to operate on.
    :param slice_border_port: node output port that provides "starts"/"ends" values for the Slice.
    :param shape: input shape of the Slice
    :param axes: axes that "starts" and "ends" apply to
    :param node_name: Slice node name
    :return: Concat node that forms "begin"/"end" values for the StridedSlice
    """
    # the value for 'starts' or 'ends' might be maximum/minimum possible value of int64. This
    # value must be converted to maximum/minimum of int32 because such big values do not fit into the int32 which is
    # supported by the StridedSlice layer
    clamp = create_op_with_const_inputs(
        graph, Clamp, port_value_dict={1: np.iinfo(np.int32).min, 2: np.iinfo(np.int32).max},
        op_attrs=dict(name=node_name + '/Clamp'))
    clamp.in_port(0).connect(slice_border_port)
    # we have to convert "starts"/"ends" values from the network to one data type with constant values that are created
    # here to prevent type errors in Concat node
    cast = Cast(graph, dict(name=node_name + '/CastToI64', dst_type=np.int64)).create_node()
    cast.in_port(0).connect(clamp.out_port(0))
    concat = Concat(graph, dict(name=node_name + '/Concat', axis=0)).create_node()
    for value_idx, port_idx in enumerate(axes):
        concat.add_input_port(port_idx)
        # "axes" may not be sorted, so we need to split "starts"/"ends" values and connect each value to the correct
        # Concat input port
        value = create_op_with_const_inputs(
            graph, Gather, port_value_dict={1: int64_array([value_idx]), 2: int64_array(0)},
            op_attrs={'name': node_name + '/Gather'})
        cast.out_port(0).connect(value.in_port(0))
        value.out_port(0).connect(concat.in_port(port_idx))
    for port_idx in range(len(shape)):
        if not concat.is_in_port_connected(port_idx):
            concat.add_input_port(port_idx)
            # This border value would be ignored in StridedSlice because of the begin_mask\end_mask
            const = Const(graph, dict(name=node_name + '/Const', value=int64_array([0]))).create_node()
            const.out_port(0).connect(concat.in_port(port_idx))

    return concat


class ConvertSlice(MiddleReplacementPattern):
    """
    This class converts a Slice operation to StridedSlice in reshape-able way by parsing the 'starts' and 'ends'
    parameters based on the 'axes' parameter
    """

    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Slice'):
            node_name = node.soft_get('name', node.id)

            input_shape = node.in_port(0).data.get_shape()
            if node.is_in_port_connected(3):
                axes = node.in_port(3).data.get_value().copy()
                assert axes is not None, 'The input with axes is not constant for node {}'.format(node_name)
                for i, val in enumerate(axes):
                    axes[i] = get_canonical_axis_index(input_shape, val)
            else:
                axes = int64_array(range(len(input_shape)))

            ss_begin = create_ss_interval_border(graph, node.in_port(1).get_source(), input_shape, axes, node_name)
            ss_end = create_ss_interval_border(graph, node.in_port(2).get_source(), input_shape, axes, node_name)
            node.in_port(1).disconnect()
            node.in_port(2).disconnect()
            rename_nodes([(ss_begin, node_name + '/Begin'), (ss_end, node_name + '/End')])

            if node.is_in_port_connected(4):
                steps = node.in_port(4).data.get_value()
                assert steps is not None, 'The input with steps is not constant for node {}'.format(node_name)
            else:
                steps = np.ones([axes.size], dtype=np.int64)

            ss_begin_mask = np.zeros(len(input_shape), dtype=np.int64)
            ss_end_mask = np.zeros(len(input_shape), dtype=np.int64)
            ss_step = np.ones(len(input_shape), dtype=np.int64)

            for i, axis in enumerate(axes):
                ss_begin_mask[axis] = 1
                ss_end_mask[axis] = 1
                ss_step[axis] = steps[i]

            ss_strides = Const(graph, dict(name=node_name + '/Strides', value=ss_step)).create_node()

            ss = StridedSlice(graph, dict(name='ss', new_axis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                          shrink_axis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                          ellipsis_mask=np.zeros(len(input_shape), dtype=np.int64),
                                          begin_mask=ss_begin_mask,
                                          end_mask=ss_end_mask)).create_node()

            node.in_port(0).get_connection().set_destination(ss.in_port(0))
            ss.in_port(1).connect(ss_begin.out_port(0))
            ss.in_port(2).connect(ss_end.out_port(0))
            ss.in_port(3).connect(ss_strides.out_port(0))
            node.out_port(0).get_connection().set_source(ss.out_port(0))

            rename_nodes([(node, node_name + '/ShouldBeDeleted'), (ss, node_name)])
