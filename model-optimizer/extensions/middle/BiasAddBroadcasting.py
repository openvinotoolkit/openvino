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
from extensions.middle.EltwiseChecker import EltwiseChecker
from extensions.ops.elementwise import Add
from mo.front.common.layout import get_features_dim
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze
import numpy as np


class BiasAddInputBroadcasting(MiddleReplacementPattern):
    """
    In TF BiasAdd op have 2 inputs: data tensor and bias tensor. Bias always has 1D shape and should be broadcasted
    to data tensor by features dimension.

    Also replacing BiasAdd by usual Add op after broadcasting.
    """
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [EltwiseChecker]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('BiasAdd', dict(kind='op', op='Add', type='BiasAdd'))
            ],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        bias_add = match['BiasAdd']

        # Replace BiasAdd by Add operation
        new_add = Add(graph, {'name': bias_add.id + '/Add'}).create_node()

        bias_add.in_port(0).get_connection().set_destination(new_add.in_port(0))
        bias_add.in_port(1).get_connection().set_destination(new_add.in_port(1))
        bias_add.out_port(0).get_connection().set_source(new_add.out_port(0))

        if bias_add.data_format != 'NCHW':
            return

        input_shape = new_add.in_port(0).data.get_shape()
        bias_shape = new_add.in_port(1).data.get_shape()
        assert len(bias_shape) == 1

        unsqueeze_dims = np.arange(len(input_shape))
        channel_dim = get_features_dim('NCHW', len(input_shape))
        unsqueeze_dims = np.delete(unsqueeze_dims, channel_dim, 0)

        unsqueeze_node = Unsqueeze(graph, {'name': new_add.id + '/BiasUnsqueeze'}).create_node()
        unsqueeze_dims_node = Const(graph, {'name': new_add.id + '/Dims',
                                            'value': unsqueeze_dims}).create_node()
        # Reconnecting nodes
        unsqueeze_node.in_port(1).connect(unsqueeze_dims_node.out_port(0))
        unsqueeze_node['override_output_shape'] = True

        new_add.in_port(1).get_connection().insert_node(unsqueeze_node)
