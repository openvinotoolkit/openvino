"""
 Copyright (c) 2018-2019 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class NormalizeFullyConnected(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'onnx']

    def run_after(self):
        from extensions.middle.GemmToFullyConnected import GemmToFullyConnected
        return [GemmToFullyConnected]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('fc', dict(kind='op', type=lambda x: x in ['MatMul', 'FullyConnected'])),
                ('fc_output', dict(kind='data'))],
            edges=[('fc', 'fc_output')],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        """
            This pass normalize FC layer
            Example:

            (2,16,512)-->FC->(2,16,101)    =>    (2,16,512)-->Reshape-->(32,512)-->FC-->(32,101)-->Reshape-->(2,16,101)

        """
        fc = match['fc']
        fc_weights = fc.in_node(1)
        fc_output = match['fc_output']
        fc_input = fc.in_node()

        if not fc_weights.has_valid('input_channel_dim') or not fc.has_valid('out-size'):
            return

        input_shape = fc.in_node().shape
        if len(input_shape) <= 2 or np.prod(fc_input.shape[1:]) == fc_weights.shape[fc_weights.input_channel_dim]:
            return

        # Insert Reshape to normalize input for FC layer that should be in [N,C] layout
        first_reshape_shape = np.array([np.prod(input_shape[0:-1]), input_shape[-1]], dtype=np.int64)
        second_reshape_shape = np.array([*input_shape[0:-1], fc['out-size']], dtype=np.int64)
        fc_out_shape = np.array([np.prod(input_shape[0:-1]), fc['out-size']], dtype=np.int64)

        first_reshape = create_op_node_with_second_input(graph, Reshape, int64_array(first_reshape_shape),
                                                         {'name': fc.name + '/Reshape'})
        fc.in_port(0).get_connection().insert_node(first_reshape)

        second_reshape = create_op_node_with_second_input(graph, Reshape, int64_array(second_reshape_shape),
                                                          {'name': fc.name + '/ReshapeBack'})

        fc.out_port(0).get_connection().insert_node(second_reshape)
        fc.out_port(0).data.set_shape(fc_out_shape)

        # run shape inference to overwrite shapes
        first_reshape.in_port(1).get_source().node.infer(first_reshape.in_port(1).get_source().node)
        first_reshape.infer(first_reshape)
