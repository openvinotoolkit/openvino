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
from mo.graph.perm_inputs import PermuteInputs
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.op import PermuteAttrs


class StridedSliceNormalizer(MiddleReplacementPattern):
    """
    StridedSlice is not normal if it cannot be permuted by ApplyPermutations. This normalizer
    inserts blank colons ':' in slice expression so that it can be correctly permuted
    from NHWC to NCHW layout. It changes masks and inserts blank begin, end and strides values.
    In order to successfully run in ShapeOf subgraphs insertations  must be done by inserting nodes
    not just by rewritting constants.

    StridedSlice is not normal in 2 cases:
        1. rank of a slice expression is less than rank of input tensor
        2. there is an ellipsis

    1st case example
    BEFORE:
                  |
                begin
           value=[0, 0]
                  |

    AFTER:
                  |
                begin          Const
           value=[0, 0]     value=[0, 0]
                  \             /
                   \           /
                      Concat
                 value=[0, 0, 0, 0]
                        |

    Input of a shape [16, 100, 100, 3] in NHWC layout, output = input[:, 0:50] will be extended to input[:, 0:50, :, :]
    after permutation to NCHW output = input[:, :, 0:50, :]. Above is show only for begin input, for end and strides
    changes are analogously.

    2nd case example
    BEFORE:
                  |
                begin
           value=[1, 50]
                  |

    AFTER:
                  |
                begin
           value=[1, 1, 1]
                  |
             VariadicSplit
           /              \
          /                \
         /       Const      \
         \     val=[0, 0]   /
          \       |        /
           \      |       /
               Concat
           value=[1, 0, 0, 1, 1]
                  |

    Input of a shape [16, 10, 100, 100, 3] in NDHWC layout output = input[1:4, ..., 1:51, 1:3],
    output_shape = [3, 10, 100, 50, 2]. In order to do correctly layout permutation in slice expression
    input[1:4, ..., 1:51, 1:3] ellipsis should be exended => input[1:4, :, :, 1:51, 1:3]. After
    layour permutation input[1:4, 1:3, :, : 1:5].

    In the places of colons blank zero begin, end and strides values
    should be inserted. In order to do that we split begin, and concatenate with the blank zeros in the middle. Above
    is show only for begin input, for end and strides changes are analogously.
    """
    enabled = True

    def run_before(self):
        from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
        return [LayoutChangeForConstantShapePaths]

    def find_and_replace_pattern(self, graph: Graph):
        ss_nodes = graph.get_op_nodes(op='StridedSlice')
        for node in ss_nodes:
            self.normalize_strided_slice(graph, node)
            PermuteAttrs.create_permute_attrs(node, attrs=[('begin_mask', 'input:0'),  # but indeed depends from slice_rank
                                                           ('end_mask', 'input:0'),
                                                           ('new_axis_mask', 'input:0'),
                                                           ('shrink_axis_mask', 'input:0'),
                                                           ('ellipsis_mask', 'input:0')])

            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:1', 'slice', 'dim_size')
            PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:2', 'slice', 'dim_size')
            PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:3', 'slice', 'dim_size')

    def normalize_strided_slice(self, graph: Graph, node: Node):
        input_shape = node.in_port(0).data.get_shape()
        input_rank = len(input_shape)
        slice_rank = node.in_port(1).data.get_shape()[0]

        slice_mask_names = ['begin_mask', 'end_mask', 'new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask']
        # align masks sizes with slice_rank (not confuse with extending mask_aligment != mask_extending)
        for mask_name in slice_mask_names:
            num = slice_rank - len(node[mask_name])
            val = 0 if mask_name not in ['begin_mask', 'end_mask'] else 1  # extend with ones only for begin and end
            node[mask_name] = np.append(node[mask_name], [val] * num).astype(int)

        if np.any(node.ellipsis_mask):
            idx = np.nonzero(node.ellipsis_mask)
            assert len(idx[0]) == 1, 'only one ellipsis_mask nonzero value is allowed'
            ellipsis_start = idx[0][0]
            # since we don't expect values in begin and end: take the whole range along ellipsis_start
            node.begin_mask[ellipsis_start] = 0
            node.end_mask[ellipsis_start] = 0

            num = input_rank - slice_rank + np.count_nonzero(node.new_axis_mask)

            # unroll ellipsis for masks
            node.ellipsis_mask[ellipsis_start] = 0
            for mask_name in slice_mask_names:
                node[mask_name] = np.insert(node[mask_name], ellipsis_start + 1, [0] * num).astype(int)

            self.unroll_ellipsis_for_inputs(graph, node, ellipsis_start, num)
        elif slice_rank - np.count_nonzero(node.new_axis_mask) < input_rank:
            num = input_rank - slice_rank + np.count_nonzero(node.new_axis_mask)
            self.extend_inputs(node, num)

            # extend masks
            for mask_name in slice_mask_names:
                num = input_rank - len(node[mask_name]) + np.count_nonzero(node.new_axis_mask)
                node[mask_name] = np.append(node[mask_name], [0] * num).astype(int)

    @staticmethod
    def unroll_ellipsis_for_inputs(graph: Graph, node: Node, ellipsis_start: int, num_ellipsis_ext: int):
        node_name = node.soft_get('name', node.id)
        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            if ellipsis_start != 0:
                split = create_op_with_const_inputs(graph, VariadicSplit, {1: 0, 2: int64_array([ellipsis_start, -1])},
                                                    {'name': node_name + '/split_to_unroll_{}_ellipsis'.format(slice_name),
                                                     'out_ports_count': 2})
                node.in_port(i).get_connection().set_destination(split.in_port(0))

            placeholder_arr = np.zeros(num_ellipsis_ext) if i != 3 else np.ones(num_ellipsis_ext)
            placeholder_node = Const(graph, {'name': node_name + '/const_to_unroll_{}_ellipsis'.format(slice_name),
                                             'value': int64_array(placeholder_arr)}).create_node()
            in_ports = 3 if ellipsis_start != 0 else 2
            concat = Concat(graph, {'axis': 0, 'name': node_name + '/concat_{}'.format(slice_name),
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
        graph = node.graph
        node_name = node.soft_get('name', node.id)

        for i, slice_name in enumerate(('begin', 'end', 'strides')):
            i += 1
            placeholder_arr = np.zeros(num) if i != 3 else np.ones(num)
            placeholder_node = Const(graph, {'name': node_name + '/extend_{}_const'.format(slice_name),
                                             'value': int64_array(placeholder_arr)}).create_node()

            if node.in_port(i).get_source().node.soft_get('type') == 'Concat':
                # concat already exists
                concat = node.in_port(i).get_source().node
                last_in_port = concat['in_ports_count']
                assert not concat.in_port(last_in_port - 1).disconnected(), 'The last in_port of Concat node {}' \
                                                                            'should be connected'.\
                    format(concat.soft_get('name', node.id))

                concat.add_input_port(last_in_port)
                concat.in_port(last_in_port).connect(placeholder_node.out_port(0))
            else:
                # have to create concat
                concat = Concat(graph, {'axis': 0, 'name': node_name + '/concat_{}'.format(slice_name),
                                        'in_ports_count': 2}).create_node()
                node.in_port(i).get_connection().set_destination(concat.in_port(0))
                concat.in_port(1).connect(placeholder_node.out_port(0))
                concat.out_port(0).get_connection().set_destination(node.in_port(i))
