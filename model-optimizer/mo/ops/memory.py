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

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class Memory(Op):
    op = 'Memory'
    enabled = True

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'type': 'Memory',
            'op': 'Memory',
            'id': None,
            'size': None,
            'index': None,
            'infer': Memory.infer,
        }, attrs)

    def supported_attrs(self):
        return ['id', 'size', 'index']

    @staticmethod
    def infer(node: Node):
        outn = node.out_node(0)  # data
        if len(node.in_nodes()) > 0:
            # In case this is a memory node with input,
            # It should not have output
            # However in order not to break MO pipeline,
            # we just set the same shape to the output
            # node that will be removed later in pipeline
            copy_shape_infer(node)
            return
        data_outs = outn.out_nodes()  # children
        for out in data_outs:
            if len(out.pb.blobs) == 0 or not isinstance(out.pb.blobs[0], np.ndarray):
                continue
            blob_shape = out.pb.blobs[0].shape[0]
            if out.type == 'FullyConnected':
                outn.shape = np.int64(np.array([1, blob_shape / out.pb.num_output]))
                break
            elif out.type == 'ScaleShift':
                outn.shape = np.int64(np.array([1, blob_shape]))
                break
        else:
            raise Error('Model Optimizer is unable to calculate output shape of Memory node {}. ' +
                        refer_to_faq_msg(88),
                        node.id)
