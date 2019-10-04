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

import logging as log

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class Slice(Op):
    op = 'Slice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': 'Slice',
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    def supported_attrs(self):
        return ['start', 'end', 'axis']

    @staticmethod
    def infer(node: Node):
        axis = None
        steps = None
        if len(node.in_nodes()) == 1:
            # Caffe or ONNX before 10 opset
            if node.has('start') and node.has('end') and node.has('axis'):
                # ONNX case
                if node.has_valid('start') and node.has_valid('end') and node.has('axis'):
                    start = node.start
                    end = node.end
                    axis = node.axis
                else:
                    log.warning('Incorrect slice operation: no starts or end attr')
                    return
            else:
                # Caffe case
                from mo.front.common.partial_infer.slice import caffe_slice_infer
                caffe_slice_infer(node)
        elif len(node.in_nodes()) >= 3:
            if node.has('format') and node['format'] == 'onnx':
                # ONNX 10 opset case
                starts_node = node.in_node(1)
                ends_node = node.in_node(2)
                if starts_node.has_valid('value') and ends_node.has_valid('value'):
                    start = np.array(node.in_node(1).value, dtype=np.int64)
                    end = np.array(node.in_node(2).value, dtype=np.int64)
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
                else:
                    log.warning('Incorrect slice operation: no starts or ends attr')
                    return
            else:
                # TF case
                start_node = node.in_node(1)
                size_node = node.in_node(2)
                if start_node.has_valid('value') and size_node.has_valid('value'):
                    start = np.array(node.in_node(1).value, dtype=np.int64)
                    size = np.array(node.in_node(2).value, dtype=np.int64)
                    end = start + size
                    axis = None

                    # Delete edges to start, size nodes
                    node.graph.remove_edge(node.in_node(1).id, node.id)
                    node.graph.remove_edge(node.in_node(2).id, node.id)

                    node['start'] = start
                    node['end'] = end
                    node['axis'] = None
                else:
                    log.warning('Incorrect slice operation: no starts or end attr')
                    return
        else:
            log.warning('Incorrect number of input nodes in slice operation')
            return

        input_shape = node.in_node(0).shape
        # Check for situation when size[i] == -1 in TF
        for i in range(start.size):
            if end[i] < start[i]:
                end[i] = input_shape[i]
        # Update end param
        node.end = end
        value = node.in_node(0).value

        # If value is None create dummy vaue for shape propogation
        if value is None:
            value = np.zeros(input_shape)

        # Following ONNX and TF specification, in case of unknown axis, axises should be in greater order
        if axis is None:
            axis = [x for x in range(len(start))]

        if steps is None:
            steps = np.ones(start.size, dtype=np.int64)

        # Calculate output value for slice operation
        slice_idx = [None for x in range(len(node.in_node().shape))]
        shrink_axis_mask = [False for x in range(len(node.in_node().shape))]
        for id in range(len(axis)):
            # Ranged for output value for specified axis
            slice_idx[axis[id]] = slice(start[id], end[id], steps[id])

        # TODO: check whether this check is really important
        for axis, s in enumerate(slice_idx):
            if s is None:
                slice_idx[axis] = slice(0, input_shape[axis], 1)

        # Add new parameters to node
        node['slices'] = np.array(slice_idx)
        node['shrink_axis_mask'] = np.array(shrink_axis_mask)

        value = value[tuple(slice_idx)]
        node.out_node().value = value.copy() if node.in_node(0).value is not None else None
        node.out_node().shape = np.array(value.shape)
