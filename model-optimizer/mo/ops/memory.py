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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class Memory(Op):
    op = 'Memory'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': 'Memory',
            'op': 'Memory',
            'id': None,
            'size': None,
            'index': None,
            'infer': Memory.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['id', 'size', 'index']

    @staticmethod
    def infer(node: Node):
        if len(node.in_nodes()) > 0:
            # In case this is a memory node with input,
            # It should not have output
            # However in order not to break MO pipeline,
            # we just set the same shape to the output
            # node that will be removed later in pipeline
            copy_shape_infer(node)
            return
        elif node.has_valid('shape'):
            # For Memories, that has not input infer shapes is very difficult
            # But often we can know shape in extracting attributes
            # And we can set the attribute 'shape' in extracting
            batch = 1
            for out_node in node.out_nodes().values():
                out_node.shape = [batch, *node.shape[:]]
            return
        else:
            raise Error('Model Optimizer is unable to calculate output shape of Memory node {}. ' +
                        refer_to_faq_msg(88),
                        node.id)
