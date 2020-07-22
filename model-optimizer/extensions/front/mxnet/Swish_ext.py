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

from extensions.ops.Swish import Swish
from mo.front.common.replacement import FrontReplacementOp
from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Node, Graph


class SwishFrontExtractor(FrontReplacementOp):
    op = 'Swish'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):

        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        beta = attrs.float('beta', 1.0)

        if beta != 1.0:
            swish = create_op_node_with_second_input(graph, Swish, np.array(beta), {'name': node.name})
        else:
            swish = Swish(graph, {'name': node.name})

        node.in_port(0).get_connection().set_destination(swish.in_port(0))
        return [swish.id]
