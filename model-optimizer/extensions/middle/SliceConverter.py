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

from extensions.ops.Cast import Cast
from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.graph.port import Port
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.strided_slice import StridedSlice


def create_ss_interval_border(graph: Graph, shape, axes, port_to_connect: Port, node_name):
    shape_mask = np.zeros(len(shape), dtype=np.int64)
    first_part = shape_mask[:axes[0]]
    last_part = shape_mask[axes[-1] + 1:]

    cast = Cast(graph, dict(name=node_name + '/CastToI64', dst_type=np.int64)).create_node()
    port_to_connect.get_connection().set_destination(cast.in_port(0))
    concat = create_op_with_const_inputs(graph, Concat, port_value_dict={0: first_part, 2: last_part},
                                         op_attrs={'name': node_name + '/Concat', 'axis': 0,
                                                   'in_ports_count': 3})
    cast.out_port(0).connect(concat.in_port(1))
    return concat


class ConvertSlice(MiddleReplacementPattern):
    """
    This class converts Slice operation to StridedSlice
    """

    enabled = True
    force_clean_up = True
    op = "Slice"

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

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

            ss_begin = create_ss_interval_border(graph, input_shape, axes, node.in_port(1).get_source(), node_name)
            ss_end = create_ss_interval_border(graph, input_shape, axes, node.in_port(2).get_source(), node_name)
            rename_nodes([(ss_begin, node_name + '/Begin'), (ss_end, node_name + '/End')])

            if node.is_in_port_connected(4):
                steps = node.in_port(4).data.get_value()
                assert steps is not None, 'The input with steps is not constant for node {}'.format(node_name)
            else:
                steps = np.ones([axes.size])

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
                                          end_mask=ss_end_mask, override_output_shape=True)).create_node()

            node.in_port(0).get_connection().set_destination(ss.in_port(0))
            ss.in_port(1).connect(ss_begin.out_port(0))
            ss.in_port(2).connect(ss_end.out_port(0))
            ss.in_port(3).connect(ss_strides.out_port(0))
            node.out_port(0).get_connection().set_source(ss.out_port(0))

            rename_nodes([(node, node_name + '/ShouldBeDeleted'), (ss, node_name)])
