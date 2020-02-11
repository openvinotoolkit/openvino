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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class SwapDeconvInputs(FrontReplacementSubgraph):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(swap_0_and_2_inputs=True):
            shape_src = node.in_port(0).get_source()
            node.in_port(0).disconnect()

            node.in_port(2).get_connection().set_destination(node.in_port(0))
            shape_src.connect(node.in_port(2))
            node['swap_0_and_2_inputs'] = False
