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

import copy

import networkx as nx
import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op
import logging as log


class Slice(Op):
    op = 'Slice'
    enabled = True

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        super().__init__(graph, {
            'op': 'Slice',
            'infer': __class__.infer
        }, attrs)

    @staticmethod
    def infer(node: Node):
        if node.start is None or node.end is None:
            log.warning('Incorrect slice operation: no starts or ends attr')
            return

        if len(node.in_nodes()) != 1:
            log.warning('Incorrect slice operation: slice op should have exactly one input')
            return

        if node.in_node(0).value is None:
            log.info('Slice operation supports only on constant path')
            return

        axis = node.axis
        start = node.start
        end = node.end
        value = node.in_node(0).value
        input_shape = node.in_node(0).shape

        # Following ONNX specification, in case of unknown axis, axises should be in greater order
        if axis is None:
            axis = [x for x in range(len(start))]

        # Calculate output value for slice operation
        slice_idx = [None for x in range(len(axis))]
        for id in range(len(axis)):
            # Ranged for output value for specified axis
            slice_idx[axis[id]] = slice(start[id], end[id], 1)

        for axis, s in enumerate(slice_idx):
            if s is None:
                slice_idx[axis] = slice(0, input_shape[axis], 1)

        value = value[slice_idx]
        node.out_node().value = np.array(value)
        node.out_node().shape = np.array(value.shape)
