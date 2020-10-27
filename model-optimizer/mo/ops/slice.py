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
from typing import List

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error

"""
Slicing operations have different semantic or different parameters/inputs in different frameworks. To distinguish them 
several internal operations are introduced. The internal MO Slice operation behaves same as Slice in ONNX opset >= 10. 
A number of transformations take place on the front phase to convert framework slicing:
 - AttributedSlice, TFSlice -> Slice 
 - CaffeSlice -> Split 
 - MXSlice -> StridedSlice 
"""


class AttributedSlice(Op):
    """
    AttributedSlice is used in old versions of ONNX models (opset version < 10).
    The operation is replaced with the OpenVINO Slice operation on the front phase.
    """
    op = 'AttributedSlice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)


class CaffeSlice(Op):
    """
    Slice in Caffe is equivalent to Split operation in OpenVINO.
    https://caffe.berkeleyvision.org/tutorial/layers/slice.html
    The operation is replaced with the OpenVINO Split operation on the front phase.
    """
    op = 'CaffeSlice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)



class TFSlice(Op):
    """
    Slice operation in Tensorflow is different from Slice in ONNX, Caffe and MXNet. It has begin and size inputs while
    ONNX Slice and internal MO Slice has start, end, step and axis parameters specified as inputs.
    https://www.tensorflow.org/api_docs/python/tf/slice
    The operation is replaced with the internal Slice on the front phase.
    If size[i] == -1 is replaced to int32_max value for the end.
    """
    op = 'TFSlice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)


class MXSlice(Op):
    """
    Slice operation in MXNet is different from ONNX, Caffe, Tensorflow. It has begin, end & step attributes
    https://mxnet.apache.org/versions/1.6/api/python/docs/api/symbol/op/index.html#mxnet.symbol.op.slice
    The operation is replaced with the OpenVINO StridedSlice operation on the front phase.
    """
    op = 'MXSlice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': None,
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': None
        }, attrs)


class Slice(Op):
    """
    Semantic of MO internal Slice operation is identical to Slice in ONNX opset >= 10.
    It has starts, ends, steps and axes inputs.
    The operation is internal (not present in the OpenVINO opset) and is replaced to StridedSlice.
    """
    op = 'Slice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict = None):
        super().__init__(graph, {
            'type': None,
            'op': 'Slice',
            'in_ports_count': 5,
            'out_ports_count': 1,
            'infer': self.infer
        }, attrs)

    @staticmethod
    def infer(node: Node):
        input_value = node.in_port(0).data.get_value()
        input_shape = node.in_port(0).data.get_shape()

        starts = node.in_port(1).data.get_value()
        ends = node.in_port(2).data.get_value()
        if starts is None or ends is None:
            raise Error('The non-constant start/end values for Slice operation "{}" are not supported'.format(node.name))

        if node.is_in_port_connected(3):
            axes = node.in_port(3).data.get_value()
            if axes is None:
                raise Error('The non-constant axes values for Slice operation "{}" is not supported'.format(node.name))
        else:
            axes = [x for x in range(len(starts))]

        if node.is_in_port_connected(4):
            steps = node.in_port(4).data.get_value()
            if steps is None:
                raise Error('The non-constant steps values for Slice operation "{}" is not supported'.format(node.name))
        else:
            steps = np.ones(len(starts), dtype=np.int64)

        slice_idx = [slice(0, in_shape, 1) for in_shape in input_shape]
        for i in range(len(axes)):
            # Ranged for output value for specified axis
            slice_idx[axes[i]] = slice(starts[i], ends[i], steps[i])
        if input_value is None:
            output_shape = get_shape_after_slice(input_shape, slice_idx)
            if np.any(output_shape <= 0):
                raise Error('Output shape: {} of node "{}" contains non-positive values'.format(output_shape, node.name))
            node.out_port(0).data.set_shape(output_shape)
        else:
            node.out_port(0).data.set_value(input_value[tuple(slice_idx)])


def get_shape_after_slice(input_shape: np.ndarray, slice_idx: List[slice]) -> np.ndarray:
    """
    Calculate shape of a tensor after slicing without actually creating the resulting tensor.
    Is introduced to prevent potentially large memory consumption.
    """
    output_shape = np.zeros(len(input_shape), dtype=np.int32)
    for i, s in enumerate(slice_idx):
        output_shape[i] = len(range(*s.indices(input_shape[i])))
    return output_shape
