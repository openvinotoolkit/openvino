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

from extensions.ops.hard_sigmoid import HardSigmoid
from mo.front.common.replacement import FrontReplacementOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class HardSigmoidFrontExtractor(FrontReplacementOp):
    op = 'HardSigmoid'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        alpha = onnx_attr(node, 'alpha', 'f', default=0.2)
        beta = onnx_attr(node, 'beta', 'f', default=0.5)
        alpha_node = Const(graph, {'value': np.array(alpha)}).create_node()
        beta_node = Const(graph, {'value': np.array(beta)}).create_node()

        hard_sigmoid = HardSigmoid(graph, {'name': node.name + '/HardSigmoid_'}).create_node()
        node.in_port(0).get_connection().set_destination(hard_sigmoid.in_port(0))
        alpha_node.out_port(0).connect(hard_sigmoid.in_port(1))
        beta_node.out_port(0).connect(hard_sigmoid.in_port(2))
        return [hard_sigmoid.id]
