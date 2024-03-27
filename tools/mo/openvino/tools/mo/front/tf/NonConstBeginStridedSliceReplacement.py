# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class NonConstBeginStridedSliceReplacement(FrontReplacementSubgraph):
    r"""
    The transformation handles StridedSlice operation with dynamic begin and end values
    when slicing performs along just one dimension with a dynamic index.
    For example, StridedSlice with begin=(0,idx,0), end=(0,idx+1,0),
    and begin_mask=end_mask=shrink_mask=(0, 1, 0) can be replaced with Squeeze(axis=1;Gather(axis=1; Unsqueeze(idx))).
    The transformation attempts to match to following sub-graph:

    Input ----> StridedSlice(begin_mask, end_mask, and shrink_mask where only element for axis equal to 1) --> OTHER OPS
                   /\                  /\                  /\
                   |                   |                   |
      ---------> Pack(Begin)          Pack(End)         Const(Step) = (1,..,1)
      |               /\                /\   /\
      |               |                 |    |
      |               |                 |   Const(All others)
      |        Const(All others)        |
    Index ---------------------------> Add
                                        /\
    Const(SliceSize)=1------------------|

    And the original sub-graph is transformed as follows:

    Input --------> Gather(axis) ---> Squeeze(axis) ---> OTHER OPS
                       /\
                       |
    Index -----> Unsqueeze(axis=1)

    """
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.Pack import Pack
        return [Pack]

    @staticmethod
    def pattern(**kwargs):
        return dict(
            nodes=[('begin', dict(op='Pack')),
                   ('end', dict(op='Pack')),
                   ('step', dict(op='Const')),
                   ('strided_slice', dict(op='StridedSlice')),
                   ],
            edges=[('begin', 'strided_slice', {'in': 1}),
                   ('end', 'strided_slice', {'in': 2}),
                   ('step', 'strided_slice', {'in': 3})])

    def replace_sub_graph(self, graph: Graph, match: dict):
        strided_slice_node = match['strided_slice']
        begin_node = match['begin']
        end_node = match['end']
        step_node = match['step']

        # retrieve attribute values
        begin_mask = strided_slice_node.soft_get('begin_mask')
        end_mask = strided_slice_node.soft_get('end_mask')
        shrink_mask = strided_slice_node.soft_get('shrink_axis_mask', int64_array([0]))

        # check applicability of this transformation to the given sub-graph:
        # 1. check that slicing is performed along just one axis
        if np.sum(begin_mask) != 1 or np.sum(end_mask) != 1 or np.sum(shrink_mask) != 1:
            return
        # 2. check that shrink axis is equal to slicing axis
        if not np.array_equal(np.argwhere(begin_mask == 1), np.argwhere(end_mask == 1)) or \
                not np.array_equal(np.argwhere(begin_mask == 1), np.argwhere(shrink_mask == 1)):
            return
        sliced_axis = np.argwhere(begin_mask == 1)[0][0]
        # 3. check constant nodes for begin and end correspond to non-slicing axes
        for idx_port, in_port in begin_node.in_ports().items():
            if idx_port != sliced_axis and in_port.get_source().node.soft_get('type') != 'Const' or \
                    idx_port == sliced_axis and in_port.get_source().node.soft_get('type') == 'Const':
                return
        for idx_port, in_port in end_node.in_ports().items():
            if idx_port != sliced_axis and in_port.get_source().node.soft_get('type') != 'Const' or \
                    idx_port == sliced_axis and in_port.get_source().node.soft_get('type') == 'Const':
                return
        # 4. check that offset of begin and end values for slicing axis is constant
        add_node = end_node.in_port(sliced_axis).get_source().node
        slice_start_index_node = begin_node.in_port(sliced_axis).get_source().node
        if add_node.soft_get('type') != 'Add':
            return

        if add_node.in_port(1).get_source().node.soft_get('type') == 'Const':
            slice_size_node = add_node.in_port(1).get_source().node
            if add_node.in_port(0).get_source().node.id != slice_start_index_node.id:
                return
        elif add_node.in_port(0).get_source().node.soft_get('type') == 'Const':
            slice_size_node = add_node.in_port(0).get_source().node
            if add_node.in_port(1).get_source().node.id != slice_start_index_node.id:
                return
        else:
            return
        slice_size = slice_size_node.value
        step_value = step_node.value[sliced_axis]

        # 5. check that step_value equal to 1 and step_value equal to 1
        # TODO: support other cases when slice_size not equal to 1 and step_value not equal to 1
        if slice_size != 1 or step_value != 1:
            return

        # unsqueeze a scalar by which to slice input tensor
        strided_slice_name = strided_slice_node.soft_get('name', strided_slice_node.id)
        unsqueeze_node = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(0)},
                                                     {'name': strided_slice_name + '/Unsqueeze'})
        add_node.in_port(0).get_connection().add_destination(unsqueeze_node.in_port(0))

        # replace StridedSlice with Gather operation that supports dynamic indices for slicing
        gather_node = create_op_with_const_inputs(graph, Gather, {2: int64_array(sliced_axis)},
                                                  {'name': strided_slice_name + '/Gather'})
        strided_slice_node.in_port(0).get_connection().set_destination(gather_node.in_port(0))
        unsqueeze_node.out_port(0).connect(gather_node.in_port(1))

        # squeeze Gather output since sliced axis is shrinked
        squeeze_node = create_op_with_const_inputs(graph, Squeeze, {1: int64_array(sliced_axis)},
                                                   {'name': strided_slice_name + '/Squeeze'})
        squeeze_node.in_port(0).connect(gather_node.out_port(0))
        rename_nodes(
            [(strided_slice_node, strided_slice_name + '/AbandonedName'), (squeeze_node, strided_slice_name)])

        # preserve a name of original StridedSlice node
        strided_slice_node.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
