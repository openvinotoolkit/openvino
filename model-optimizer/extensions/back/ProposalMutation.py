"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape


class ProposalMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_before(self):
        return [ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('proposal', {'type': 'Proposal'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['proposal']
        assert len(node.in_ports()) == 3, "Proposal op must have exactly 3 input ports"
        im_info_shape = node.in_port(2).data.get_shape()
        assert im_info_shape is not None

        if np.array_equal(im_info_shape, [1, 3]) or np.array_equal(im_info_shape, [1, 4]):
            reshape = Reshape(graph, dict(name="im_info/Reshape")).create_node()
            const = Const(graph, dict(value=[im_info_shape[1]])).create_node()
            node.in_port(2).get_connection().set_destination(reshape.in_port(0))
            const.out_port(0).connect(reshape.in_port(1))
            reshape.out_port(0).connect(node.in_port(2))
