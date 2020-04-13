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

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph


class PadToV7(BackReplacementPattern):
    """
    Transformation converts Pad representation from IR V10 to IR V7.
    """
    enabled = True
    force_clean_up = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Pad'):
            pad_name = node.soft_get('name', node.id)
            pb = node.in_port(1).data.get_value()
            pe = node.in_port(2).data.get_value()

            assert pb is not None, 'Begin pads are not constants for node "{}"'.format(pad_name)
            assert pe is not None, 'End pads are not constants for node "{}"'.format(pad_name)

            node['pads'] = np.concatenate([pb.reshape([-1, 1]), pe.reshape([-1, 1])], axis=1)

            if not node.in_port(3).disconnect():
                fill_value = node.in_port(3).data.get_value()
                assert fill_value is not None, 'Fill value is not constants for node "{}"'.format(pad_name)
                node.fill_value = fill_value

            node.in_port(1).disconnect()
            node.in_port(2).disconnect()
            if not node.in_port(3).disconnected():
                node.in_port(3).disconnect()
