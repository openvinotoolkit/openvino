"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class NormalizeFullyConnected(MiddleReplacementPattern):
    enabled = False

    def pattern(self):
        return dict(
            nodes=[
                ('fc', dict(kind='op', type='FullyConnected')),
                ('fc_output', dict(kind='data'))],
            edges=[('fc', 'fc_output')],
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        """
            This pass normalize FC layer
            Example:

            (2,16,512)-->FC->(2,16,101)    =>    (2,16,512)-->Reshape-->(32,512)-->FC-->(32,101)-->Reshape-->(2,16,101)

        """
        fc = match['fc']
        fc_weights = fc.in_node(1)
        fc_output = match['fc_output']
        fc_input = fc.in_node()

        input_shape = fc.in_node().shape
        if len(input_shape) <= 2 or np.prod(fc_input.shape[1:]) == fc_weights.shape[fc_weights.input_channel_dim]:
            return

        # Insert Reshape to normalize input for FC layer that should be in [N,C] layout
        first_reshape_shape = np.array([np.prod(input_shape[0:-1]), input_shape[-1]], dtype=np.int64)
        second_reshape_shape = np.array([*input_shape[0:-1], fc['out-size']], dtype=np.int64)
        fc_out_shape = np.array([np.prod(input_shape[0:-1]), fc['out-size']], dtype=np.int64)
        first_reshape = Reshape(graph, {'dim': np.array(first_reshape_shape)})
        second_reshape = Reshape(graph, {'dim': np.array(second_reshape_shape)})

        input_edge_attrs = graph.get_edge_data(fc_input.id, fc.id)[0]
        output_edge_attrs = graph.get_edge_data(fc.id, fc_output.id)[0]

        graph.remove_edge(fc_input.id, fc.id)
        graph.remove_edge(fc.id, fc_output.id)

        # Insert Reshapes before and after FullyConnected layer
        reshape_data = first_reshape.create_node_with_data(inputs=[fc_input])
        graph.add_edge(reshape_data.id, fc.id, **input_edge_attrs)

        new_fc_output = Op.create_data_node(graph, fc, {'shape': fc_out_shape}, edge_attrs=output_edge_attrs)

        second_reshape.create_node_with_data(inputs=[new_fc_output], data_nodes=fc_output)
