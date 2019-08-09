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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from extensions.ops.elementwise import Mul
from mo.ops.op import Op
from mo.utils.error import Error


class ScaleInput(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def pattern(self):
        return dict(
            nodes=[
                ('placeholder', dict(kind='op', op='Parameter')),
                ('data', dict(kind='data'))],
            edges=[
                ('placeholder', 'data'),
            ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        scale = graph.graph['cmd_params'].scale
        if scale is None or scale == 1:
            return
        assert (len(match['placeholder'].out_nodes()))

        tinput = match['placeholder']
        if not tinput.has_valid('shape'):
            raise Error("Node {} has not valid shape attribute".format(tinput.id))

        input_shape = tinput.shape
        toutput = match['data']

        # Create Mul node
        value = np.array([1 / scale])

        # Disconnect input with data node
        graph.remove_edge(tinput.id, toutput.id)

        # Create Mul node
        mul_node = Mul(graph, dict(name="Mul1_"))
        mul_data = Op.create_input_data_node(graph, "data_mul_scale_", np.array(value))
        Op.expand_node_shape(mul_data, len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 0)
        mul_input = Op.create_data_node(graph, tinput, {'shape': toutput.shape})

        mul_node.create_node_with_data(inputs=[mul_input, mul_data], data_nodes=toutput)
