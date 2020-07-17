"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.strided_slice import StridedSlice
from mo.utils.error import Error


def convert_negative_indices(indices: np.array, shape: np.array):
    for ind, value in enumerate(indices):
        if value < 0:
            indices[ind] += shape[ind]


class ConvertSlice(MiddleReplacementPattern):
    """
    This class converts Slice operation to StridedSlice
    """

    enabled = True
    op = "Slice"
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('slice', dict(kind='op', op='Slice'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['slice']

        input_shape = node.in_port(0).data.get_shape()
        output_shape = node.out_port(0).data.get_shape()
        starts = node.in_port(1).data.get_value()
        ends = node.in_port(2).data.get_value()
        if starts is None or ends is None:
            raise Error('The input with starts or end is not constant for node {}'.format(node.id))

        # the value for 'ends' is usually maximum possible value of int64. This
        # value must be converted to maximum of int32 because such big values do not fit into the int32 which is
        # supported by the StridedSlice layer
        ends = np.clip(ends, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
        if node.is_in_port_connected(3):
            axes = node.in_port(3).data.get_value()
            if axes is None:
                raise Error('The input with axes is not constant for node {}'.format(node.id))
        else:
            axes = int64_array(list(range(starts.size)))

        if node.is_in_port_connected(4):
            steps = node.in_port(4).data.get_value()
            if steps is None:
                raise Error('The input with steps is not constant for node {}'.format(node.id))
        else:
            steps = np.ones([starts.size])

        ss_begin_mask = np.zeros(len(input_shape), dtype=np.int32)
        ss_end_mask = np.zeros(len(input_shape), dtype=np.int32)
        ss_begin = np.zeros(len(input_shape), dtype=np.int32)
        ss_end = np.zeros(len(input_shape), dtype=np.int32)
        ss_step = np.ones(len(input_shape), dtype=np.int32)

        # prepare inputs and attributes for the StridedSlice layer
        for i, axis in enumerate(axes):
            if starts[i] != 0:
                ss_begin_mask[axis] = 1
                ss_begin[axis] = starts[i]

            ss_end_mask[axis] = 1
            ss_end[axis] = ends[i]

            ss_step[axis] = steps[i]

        slice_node_name = node.soft_get('name', node.id)

        begin_node = Const(graph, {'value': ss_begin, 'name': slice_node_name + '/begin'}).create_node()
        end_node = Const(graph, {'value': ss_end, 'name': slice_node_name + '/end'}).create_node()
        strides_node = Const(graph, {'value': ss_step, 'name': slice_node_name + '/stride'}).create_node()

        ss = StridedSlice(graph, dict(new_axis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      shrink_axis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      ellipsis_mask=np.zeros(len(output_shape), dtype=np.int32),
                                      begin_mask=ss_begin_mask,
                                      end_mask=ss_end_mask)).create_node()
        rename_nodes([(node, slice_node_name + '_delete'), (ss, slice_node_name)])
        node.in_port(0).get_connection().set_destination(ss.in_port(0))
        begin_node.out_port(0).connect(ss.in_port(1))
        end_node.out_port(0).connect(ss.in_port(2))
        strides_node.out_port(0).connect(ss.in_port(3))
        node.out_port(0).get_connection().set_source(ss.out_port(0))
