# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils \
    import create_op_with_const_inputs, create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.ops.split import VariadicSplit
from openvino.tools.mo.ops.memoryoffset import MemoryOffset
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.einsum import Einsum
from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.ops.softmax import Softmax


class RestrictedAttentionComponentReplacer(FrontReplacementPattern):
    r"""
    This class expands RestrictedAttention operator into the following subgraph:

                    placeholder
                        |
            Reshape[batch*num_heads, -1]
                        |
    VariadicSplit(val_dim, key_dim, key_dim + context_dim)
                        |
              __________________________
             |           |               \
             |      MemoryOffset*     VariadicSplit(key_dim, contex_dim)
             |                   \      /      |
             |                  Einsum(dot)    |
             |                       |         |
             |                 Mul(key_scale)  |
             |                         \       |
             |                           ______
             |                             |
             |                            Add
             |                             |
      MemoryOffset*                    SoftMax
                \                     /     |
                  __________________        |
                          |                 |
                     Einsum(dot)            |
                              \            /
                                __________
                                    |
                                  Concat
                                    |
                            Reshape[batch, -1]
                                    |

    where context_dim = num_left_inputs + num_right_inputs + 1.
    *MemoryOffsets are described in the create_memory_offsets_subgraph method.
    Specification of the RestrictedAttention Kaldi operator can be found in the Kaldi documentation:
    https://kaldi-asr.org/doc/classkaldi_1_1nnet3_1_1RestrictedAttentionComponent.html.
    """
    enabled = True
    run_not_recursively = True

    def __init__(self) -> None:
        self.in_name: str
        self.num_left_inputs: int
        self.num_right_inputs: int
        self.time_stride: int
        super().__init__()

    def run_before(self):
        from openvino.tools.mo.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        return [MemoryOffsetAdjustment]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='restrictedattentioncomponent'):
            self.replace_restrictedattention(graph, node)

    def create_memory_offsets_subgraph(self, graph: Graph, input_node: Node, out_port,
                                       mem_offset_idx):
        r"""
        This method creates the following subgraph and returns endpoint Concat node:
                              input_node
                 __________________|___________________________
                /                  |              \            \
        MemoryOffset(t1)     MemoryOffset(t2)    ...    MemoryOffset(tk)
               \_____________ _____|______________/____________/
                                   |
                                 Concat
        where t1 = -time_stride*num_left_inputs, t2 = t1 + time_stride and
              tk = time_stride*num_right_inputs
        """
        concat_node = Concat(
            graph, attrs={'name': self.in_name + f'/Concat_tmp_{mem_offset_idx}'}).create_node()

        for idx, t in enumerate(list(range(-self.time_stride*self.num_left_inputs,
                                           self.time_stride*self.num_right_inputs+1)\
                                [::self.time_stride])):
            concat_node.add_input_port(idx)
            if t != 0:
                memoff = MemoryOffset(graph, attrs={'name': self.in_name +\
                                                            f'/MemoryOffset_{mem_offset_idx}_' +\
                                                            str(idx),
                                                    't': t, 'has_default': False,
                                                    'splitted': False,
                                                    'pair_name': self.in_name +
                                                    f'/MemoryOffset_{mem_offset_idx}_pair_' +
                                                    str(idx)}).create_node()
                memoff.out_port(0).connect(concat_node.in_port(idx))
                input_node.out_port(out_port).connect(memoff.in_port(0))
            else:
                # 0 time delay is not allowed in IE, it's meaningless
                # if time offset is 0 then connect input directly to Concat without memoryoffset
                input_node.out_port(out_port).connect(concat_node.in_port(idx))

        return concat_node

    def replace_restrictedattention(self, graph: Graph, restrictedattention_node: Node):
        """
        This method replaces RestrictedAttention operator with a subgraph composed with supported
        OpenVino operators.
        """

        self.num_left_inputs = restrictedattention_node['num_left_inputs']
        self.num_right_inputs = restrictedattention_node['num_right_inputs']
        context_dim = self.num_left_inputs + self.num_right_inputs + 1
        num_heads = restrictedattention_node['num_heads']
        key_dim = restrictedattention_node['key_dim']
        value_dim = restrictedattention_node['value_dim']
        self.time_stride = restrictedattention_node['time_stride']
        key_scale = restrictedattention_node['key_scale']

        batch_axis = 0
        input_shape = restrictedattention_node.in_port(0).data.get_shape()
        if input_shape:
            batch_num = input_shape[batch_axis]
        else:
            batch_num = 1

        self.in_name = restrictedattention_node.soft_get('name', restrictedattention_node.id)

        reshape_1_node = create_op_node_with_second_input(graph, Reshape,
                                                        int64_array([batch_num * num_heads, -1]),
                                                        {'name': self.in_name + '/Reshape_1'})
        restrictedattention_node.in_port(0).get_source().connect(reshape_1_node.in_port(0))

        split_1_node = create_op_with_const_inputs(graph, VariadicSplit,
                                                   {1: int64_array(1),
                                                    2: int64_array([key_dim, value_dim,
                                                                    key_dim + context_dim])},
                                                   {'name': self.in_name + '/VariadicSplit_1',
                                                   'out_ports_count': 3})
        reshape_1_node.out_port(0).connect(split_1_node.in_port(0))

        concat_1_node = self.create_memory_offsets_subgraph(graph, split_1_node, 0, 1)

        split_2_node = create_op_with_const_inputs(graph, VariadicSplit,
                                                   {1: int64_array(1),
                                                    2: int64_array([key_dim, context_dim])},
                                                   {'name': self.in_name + '/VariadicSplit_2',
                                                   'out_ports_count': 2})
        split_1_node.out_port(2).connect(split_2_node.in_port(0))

        einsum_1_node = Einsum(graph, {'name': self.in_name + '/Einsum_1',
                                       'override_output_shape': False,
                                       'in_ports_count': 2,
                                       'equation': 'ij,ik->i'}).create_node()

        reshape_helper_1_node = create_op_node_with_second_input(graph, Reshape,
                                                            int64_array(
                                                                [num_heads, 1]),
                                                            {'name': self.in_name +\
                                                                     '/Reshape_helper_1'})
        einsum_1_node.out_port(0).connect(reshape_helper_1_node.in_port(0))

        concat_1_node.out_port(0).connect(einsum_1_node.in_port(0))

        split_2_node.out_port(0).connect(einsum_1_node.in_port(1))

        mul_node = create_op_with_const_inputs(graph, Mul, {1: mo_array(key_scale, dtype=float)},
                                               {'name': self.in_name + '/Mul'})
        reshape_helper_1_node.out_port(0).connect(mul_node.in_port(0))

        add_node = Add(graph, {'name': self.in_name + '/Add'}).create_node()
        mul_node.out_port(0).connect(add_node.in_port(1))
        split_2_node.out_port(1).connect(add_node.in_port(0))

        softmax_node = Softmax(graph, {'axis': 1, 'name': self.in_name + '/Softmax'}).create_node()
        add_node.out_port(0).connect(softmax_node.in_port(0))

        concat_2_node = self.create_memory_offsets_subgraph(graph, split_1_node, 1, 2)

        reshape_helper_2_node = create_op_node_with_second_input(graph, Reshape,
                                                            int64_array([num_heads,
                                                                         value_dim,
                                                                         context_dim]),
                                                            {'name': self.in_name +\
                                                                     '/Reshape_helper_2'})
        concat_2_node.out_port(0).connect(reshape_helper_2_node.in_port(0))

        reshape_helper_3_node = create_op_node_with_second_input(graph, Reshape,
                                                            int64_array(
                                                                [num_heads, 1, context_dim]),
                                                            {'name': self.in_name +\
                                                                     '/Reshape_helper_3'})

        einsum_2_node = Einsum(graph, {'name': self.in_name + '/Einsum_2',
                                       'in_ports_count': 2,
                                       'equation': 'ijk,ilk->ij'}).create_node()
        reshape_helper_2_node.out_port(0).connect(einsum_2_node.in_port(0))

        softmax_node.out_port(0).connect(reshape_helper_3_node.in_port(0))
        reshape_helper_3_node.out_port(0).connect(einsum_2_node.in_port(1))

        concat_3_node = Concat(graph, {'name': self.in_name + '/Concat_2',
                                     'in_ports_count': 2}).create_node()
        einsum_2_node.out_port(0).connect(concat_3_node.in_port(0))
        softmax_node.out_port(0).connect(concat_3_node.in_port(1))

        reshape_2_node = create_op_node_with_second_input(graph, Reshape,
                                                          int64_array([batch_num, -1]),
                                                          {'name': self.in_name + '/Reshape_2'})
        concat_3_node.out_port(0).connect(reshape_2_node.in_port(0))

        restrictedattention_node.in_port(0).disconnect()
        restrictedattention_node.out_port(0).get_connection().set_source(reshape_2_node.out_port(0))
        graph.remove_node(restrictedattention_node.id)
