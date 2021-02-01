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
from mo.graph.perm_inputs import PermuteInputs

class StridedSliceNormalizer(MiddleReplacementPattern):
    enabled = True

    def run_before(self):
        from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
        return [LayoutChangeForConstantShapePaths]

    def find_and_replace_pattern(self, graph: Graph):
        ss_nodes = graph.get_op_nodes(op='StridedSlice')
        for node in ss_nodes:
            self.normalize_strided_slice(graph, node)
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
            PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')
            PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'shape')
            pass

    def normalize_strided_slice(self, graph: Graph, node: Node):
        input_shape = node.in_port(0).data.get_shape()
        input_rank = len(input_shape)
        slice_rank = node.in_port(1).data.get_shape()[0]

        if np.any(node.ellipsis_mask):
            idx = np.nonzero(node.ellipsis_mask)
            assert len(idx[0]) == 1
            ellipsis_start = idx[0][0]
            num_inserts = input_rank - slice_rank + np.count_nonzero(node.new_axis_mask)

            node.begin_mask[ellipsis_start] = 0
            node.end_mask[ellipsis_start] = 0

            # unroll ellipsis for masks
            for i in range(0, num_inserts):
                for mask_name in ['begin_mask', 'end_mask', 'new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask']:
                    node[mask_name] = np.insert(node[mask_name], ellipsis_start + 1 + i, 0)
            node.ellipsis_mask[ellipsis_start] = 0

            self.unroll_ellipsis_for_inputs(graph, node, ellipsis_start, num_inserts)

        elif slice_rank < input_rank:  # process somehow nonzero
            num = input_rank - (slice_rank)
            # extend masks
            for mask_name in ['begin_mask', 'end_mask', 'new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask']:
                node[mask_name] = np.append(node[mask_name], [0] * num)

            self.extend_inputs(node, num)

    @staticmethod
    def unroll_ellipsis_for_inputs(graph, node, ellipsis_start, num_ellipsis_ext):
        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            if ellipsis_start != 0:
                split = create_op_with_const_inputs(graph, VariadicSplit, {1: 0, 2: int64_array([ellipsis_start, -1])},
                                                    {'name': node.name + '/split_to_unroll_{}_ellipsis'.format(slice_name),
                                                     'out_ports_count': 2})
                node.in_port(i).get_connection().set_destination(split.in_port(0))

            placeholder_arr = np.zeros(num_ellipsis_ext) if i != 3 else np.ones(num_ellipsis_ext)
            placeholder_node = Const(graph, {'name': node.name + '/const_to_unroll_{}_ellipsis'.format(slice_name),
                                             'value': int64_array(placeholder_arr)}).create_node()
            in_ports = 3 if ellipsis_start != 0 else 2
            concat = Concat(graph, {'axis': 0, 'name': node.name + '/concat_{}'.format(slice_name),
                                    'in_ports_count': in_ports}).create_node()
            if ellipsis_start == 0:
                concat.in_port(0).connect(placeholder_node.out_port(0))
                node.in_port(i).get_connection().set_destination(concat.in_port(1))
            else:
                concat.in_port(0).connect(split.out_port(0))
                concat.in_port(1).connect(placeholder_node.out_port(0))
                concat.in_port(2).connect(split.out_port(1))

            concat.out_port(0).get_connection().set_destination(node.in_port(i))

    @staticmethod
    def extend_inputs(node: Node, num: int):
        int32_array = lambda x: np.array(x, dtype=np.int32)
        graph = node.graph

        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            placeholder_arr = np.zeros(num) if i != 3 else np.ones(num)
            placeholder_node = Const(graph, {'name': node.name + '/extend_{}_const'.format(slice_name),
                                             # 'value': int32_array(placeholder_arr)}).create_node()
                                             'value': int64_array(placeholder_arr)}).create_node()

            if node.in_port(i).get_source().node.type == 'Concat':
                # concat already exists
                concat = node.in_port(i).get_source().node
                last_in_port = concat['in_ports_count']
                assert not concat.in_port(last_in_port - 1).disconnected(), 'Previous node should be connected'
                concat.add_input_port(last_in_port)
                concat.in_port(last_in_port).connect(placeholder_node.out_port(0))
            else:
                # have to create concat
                concat = Concat(graph, {'axis': 0, 'name': node.name + '/concat_{}'.format(slice_name),
                                        'in_ports_count': 2}).create_node()
                node.in_port(i).get_connection().set_destination(concat.in_port(0))
                concat.in_port(1).connect(placeholder_node.out_port(0))
                concat.out_port(0).get_connection().set_destination(node.in_port(i))
