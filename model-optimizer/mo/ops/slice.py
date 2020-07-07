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

import logging as log

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class AttributedSlice(Op):
    op = 'AttributedSlice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)

    def supported_attrs(self):
        return ['axis', 'start', 'end']


class CaffeSlice(Op):
    op = 'CaffeSlice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)

    def supported_attrs(self):
        return ['slice_point', 'axis']

class TFSlice(Op):
    op = 'TFSlice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)

    def supported_attrs(self):
        return []

class Slice(Op):
    op = 'Slice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': 'Slice',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        value = node.in_node(0).value
        input_shape = node.in_node(0).shape
        axis = None
        steps = None

        first_input = node.in_node(1)
        second_input = node.in_node(2)
        if first_input.has_valid('value') and second_input.has_valid('value'):
            start = int64_array(first_input.value)
            end = int64_array(second_input.value)
        else:
            log.error('Incorrect slice operation: no valid starts and/or ends')
            return

        if 3 in node.in_nodes():
            if node.in_node(3).has_valid('value'):
                axis = np.array(node.in_node(3).value, dtype=np.int64)
            else:
                log.warning('Incorrect slice operation: axes should be const')
                return
        if 4 in node.in_nodes():
            if node.in_node(4).has_valid('value'):
                steps = np.array(node.in_node(4).value, dtype=np.int64)
            else:
                log.warning('Incorrect slice operation: steps should be const')
                return

        if axis is None:
            axis = [x for x in range(len(start))]

        if steps is None:
            steps = np.ones(start.size, dtype=np.int64)

        slice_idx = [None for x in range(len(input_shape))]
        for id in range(len(axis)):
            # Ranged for output value for specified axis
            slice_idx[axis[id]] = slice(start[id], end[id], steps[id])

        # this does not break reshape-ability: values along axes not touched by Slice are copied
        for axis, s in enumerate(slice_idx):
            if s is None:
                slice_idx[axis] = slice(0, input_shape[axis], 1)

        if value is None:
            value = np.zeros(input_shape)

        value = value[tuple(slice_idx)]

        node.out_node().value = value.copy() if node.in_node(0).value is not None else None
        node.out_node().shape = np.array(value.shape)
