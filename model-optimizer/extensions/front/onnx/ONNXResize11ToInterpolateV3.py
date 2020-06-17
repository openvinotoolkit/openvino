"""
 Copyright (C) 2020 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementOp
# from extensions.ops.ONNXResize11 import ONNXResize11Op
from mo.graph.graph import Graph, Node


class ONNXResize11ToInterpolate3(FrontReplacementOp):
    """
    The transformation replaces SoftPlus(x) with log(1.0 + exp(x)).
    """
    op = 'ONNXResize11'
    enabled = False

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
