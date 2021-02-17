"""
 Copyright (C) 2017-2021 Intel Corporation

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

from extensions.ops.loop import Loop
from mo.middle.replacement import MiddleReplacementPattern
from mo.graph.graph import Graph


class MoveConstToLoopBody(MiddleReplacementPattern):
    """
    It moves constant producers for Loop node into the body graph and removes input ports for them.
    This transformations helps to continue constant folding inside the body graph if possible.
    The constant folding serves as optimization path and helps to avoid issue connecting with constants
    lying on weights path to Convolution node.
    """
    enabled = True
    force_shape_inference = True

    def run_after(self):
        from extensions.middle.pass_separator import PostMiddleStart
        return [PostMiddleStart]

    def run_before(self):
        from extensions.middle.ApplyPermutations import ApplyPermutation
        return [ApplyPermutation]

    def find_and_replace_pattern(self, graph: Graph):
        cleanup_called_once = False

        # walk through all Loop nodes and find Const inputs
        for loop_node in graph.get_op_nodes(op='Loop'):
            # call clean-up only once that performs constant folding
            if not cleanup_called_once:
                graph.clean_up()
                cleanup_called_once = True

            # move constant node into the body graph and removes body parameter nodes corresponding to them
            Loop.pull_constant_inputs_into_body(loop_node)

            # since some input ports can be removed after the pulling constants, normalization of Loop node is required
            Loop.normalize_input_output_ports(loop_node)

            # perform shape inference for the Loop node again since new constant can be appeared
            # and constant folding can be helpful for weights path to Convolution node inside the body graph
            loop_node['need_shape_inference'] = True
