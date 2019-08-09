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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const
from extensions.ops.elementwise import Mul, Add


class ImageScaler(FrontReplacementOp):
    op = "ImageScaler"
    enabled = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        # This replacer replace ImageScalar operation to Mul->Add sequence
        # Also it check that weights and biases are good
        op = match['op']

        # Check that weights and biases are not useless
        has_bias, has_weights = True, True
        if all([x == 1 for x in np.nditer(op.scale)]):
            has_weights = False
        if all([x == 0 for x in np.nditer(op.bias)]):
            has_bias = False

        assert len(op.in_ports()) == 1

        last_port = op.in_port(0).get_source()

        # Create Mul & Add nodes
        if has_weights:
            mul_weights = Const(graph, dict(value=op.scale, shape=op.scale.shape)).create_node()
            mul_op = Mul(graph, dict(name=op.id + '/mul_')).create_node()
            op.in_port(0).get_connection().set_destination(mul_op.in_port(0))
            mul_weights.out_port(0).connect(mul_op.in_port(1))
            last_port = mul_op.out_port(0)

        if has_bias:
            add_bias = Const(graph, dict(value=op.bias, shape=op.bias.shape)).create_node()
            add_op = Add(graph, dict(name=op.id + '/add_')).create_node()
            last_port.get_connection().set_destination(add_op.in_port(0))
            add_bias.out_port(0).connect(add_op.in_port(1))
            last_port = add_op.out_port(0)

        op.in_port(0).disconnect()
        op.out_port(0).get_connection().set_source(last_port)
