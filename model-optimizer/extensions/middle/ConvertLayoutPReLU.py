"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern


class ConvertLayoutPReLU(MiddleReplacementPattern):
    '''
    PRelu needs to be executed in NCHW mode for input tensor of any rank
    so it injects Transpose before PRelu and after for rank not equal to 3
    '''
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for prelu in graph.get_op_nodes(op='PReLU'):
            input_shape = prelu.in_port(0).data.get_shape()
            if len(input_shape) != 3:
                continue
            output_name = prelu.soft_get('name', prelu.id)
            transpose_before = create_op_with_const_inputs(graph, Transpose, {1: int64_array([0, 2, 1])},
                                                           {'name': output_name + '/TransposeBefore'})
            prelu.in_port(0).get_connection().set_destination(transpose_before.in_port(0))
            transpose_after = create_op_with_const_inputs(graph, Transpose, {1: int64_array([0, 2, 1])},
                                                           {'name': output_name})
            rename_nodes([(prelu, output_name + '/PRelu'), (transpose_after, output_name)])

            prelu.in_port(0).connect(transpose_before.out_port(0))
            prelu.out_port(0).get_connection().set_source(transpose_after.out_port(0))
            transpose_after.in_port(0).connect(prelu.out_port(0))

            # infer shapes for newly added nodes and recalculate for PRelu node
            prelu['need_shape_inference'] = True
            transpose_before['need_shape_inference'] = True
            transpose_after['need_shape_inference'] = True
            prelu['override_output_shape'] = True
