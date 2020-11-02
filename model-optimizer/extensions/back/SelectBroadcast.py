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

from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.unsqueeze import Unsqueeze


class SelectBroadcast(BackReplacementPattern):
    """
    Select broadcasting semantics in TF isn't numpy-like
    broadcasting rules, manual reshape is needed.
    For example:
        condition: [1]
        input_1: [1, 8]
        input_2: [1, 8]
    Condition should be aligned with first dimensions of inputs.
    """
    enabled = True

    def run_before(self):
        return [ReshapeMutation]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', op='Select'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        select = match['op']
        if select.has_valid('format') and select['format'] == 'tf':
            condition = select.in_node(0)
            input_1 = select.in_node(1)
            input_2 = select.in_node(2)

            assert np.array_equal(input_1.shape, input_2.shape)

            if len(condition.shape) == 1 and len(input_1.shape) > 1:
                unsqueeze_op = create_op_node_with_second_input(
                    graph, Unsqueeze, int64_array(range(1, len(input_1.shape))),
                    {'name': select.name+'/Broadcast/'}, select.in_port(0).get_source().node)

                select.in_port(0).disconnect()
                select.in_port(0).get_connection().set_source(unsqueeze_op.out_port(0))
