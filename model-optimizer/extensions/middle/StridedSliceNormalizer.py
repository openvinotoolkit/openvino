"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.ops.split import VariadicSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const


class StridedSliceNormalizer(MiddleReplacementPattern):
    enabled = True

    def run_before(self):
        from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
        return [LayoutChangeForConstantShapePaths]

    def find_and_replace_pattern(self, graph: Graph):
        ss_nodes = graph.get_op_nodes(op='StridedSlice')
        for node in ss_nodes:
            self.normalize_strided_slice(graph, node)
            # normalize attributes


    def extend_inputs(self, graph, node, ellipsis_start, ellipsis_stop, input_rank, slice_rank):
        in_ports = 3 if ellipsis_start != 0 else 2

        num_ellipsis_ext = ellipsis_stop - ellipsis_start
        # begin, end, strides
        for i in range(1, 4):
            if ellipsis_start != 0:
                split = create_op_with_const_inputs(graph, VariadicSplit, {1: 0, 2: int64_array([ellipsis_start, -1])},
                                                    {'name': node.name + '/variadic_split_{}/'.format(i),
                                                     'out_ports_count': 2})
                node.in_port(i).get_connection().set_destination(split.in_port(0))

            placeholder_arr = np.zeros(num_ellipsis_ext) if i != 3 else np.ones(num_ellipsis_ext)
            placeholder_node = Const(graph, {'name': node.name + '/const_{}/'.format(i),
                                             'value': int64_array(placeholder_arr)}).create_node()
            concat = Concat(graph, {'axis': 0, 'name': node.name + '/concat_{}'.format(i),
                                    'in_ports_count': in_ports}).create_node()
            if ellipsis_start == 0:
                concat.in_port(0).connect(placeholder_node.out_port(0))
                node.in_port(i).get_connection().set_destination(concat.in_port(1))
            else:
                concat.in_port(0).connect(split.out_port(0))
                concat.in_port(1).connect(placeholder_node.out_port(0))
                concat.in_port(2).connect(split.out_port(1))

            concat.out_port(0).get_connection().set_destination(node.in_port(i))

    def normalize_strided_slice(self, graph: Graph, node: Node):
        input_shape = node.in_port(0).data.get_shape()
        input_rank = len(input_shape)
        slice_rank = node.in_port(1).data.get_shape()[0]
        num_ellipsis_ext = None

        if np.any(node.ellipsis_mask):
            idx = np.nonzero(node.ellipsis_mask)
            assert len(idx[0]) == 1
            ellipsis_start = idx[0][0]
            ellipsis_stop = input_rank - (slice_rank - ellipsis_start)  # stop index in extended masks

            node.begin_mask[ellipsis_start] = 0
            node.end_mask[ellipsis_start] = 0

            for i in range(ellipsis_start + 1, ellipsis_stop + 1):
                np.insert(node.begin_mask, i, 0)
                np.insert(node.end_mask, i, 0)
                np.insert(node.new_axis_mask, i, 0)
                np.insert(node.shrink_axis_mask, i, 0)
            node.ellipsis_mask[ellipsis_start] = 0
            self.extend_inputs(graph, node, ellipsis_start, ellipsis_stop, input_rank, slice_rank)
            num_ellipsis_ext = ellipsis_stop - ellipsis_start

        if num_ellipsis_ext is None:
            num_ellipsis_ext = 0

        if slice_rank + num_ellipsis_ext < input_rank:
            num = input_rank - (slice_rank + num_ellipsis_ext)
            int32_array = lambda x: np.array(x, dtype=np.int32)
            for i in range(1, 4):
                placeholder_arr = np.zeros(num) if i != 3 else np.ones(num)
                placeholder_node = Const(graph, {'name': node.name + '/const_for_endings_{}/'.format(i),
                                                 'value': int32_array(placeholder_arr)}).create_node()

                if node.in_port(i).get_source().node.type == 'Concat':
                    # concat already exists
                    concat = node.in_port(i).get_source().node
                    last_in_port = concat['in_ports_count']
                    assert not concat.in_port(last_in_port - 1).disconnected(), 'Previous node should be connected'
                    concat.add_input_port(last_in_port)
                    concat.in_port(last_in_port).connect(placeholder_node.out_port(0))
                else:
                    # have to create concat
                    concat = Concat(graph, {'axis': 0, 'name': node.name + '/concat_{}'.format(i),
                                            'in_ports_count': 2}).create_node()
                    node.in_port(i).get_connection().set_destination(concat.in_port(0))
                    concat.in_port(1).connect(placeholder_node.out_port(0))
                    concat.out_port(0).get_connection().set_destination(node.in_port(i))
