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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error

"""
Slicing operations have different semantic or different parameters/inputs in different frameworks. To distinguish them 
and not confuse with OpenVINO Slice several Model Optimizer internal operations are introduced. OpenVINO Slice operation 
behaves as ONNX Slice in opset >= 10. A number of transformations take place on the front phase to convert framework slicing 
operations to OpenVINO operations from opset:
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

    def supported_attrs(self):
        return ['axes', 'starts', 'ends']


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

    def supported_attrs(self):
        return ['slice_point', 'axis']


class TFSlice(Op):
    """
    Slice operation in Tensorflow is different from Slice in ONNX, Caffe and MXNet. It has begin and size inputs while
    OpenVINO Slice has start, end, step and axis parameters specified as inputs.
    https://www.tensorflow.org/api_docs/python/tf/slice
    The operation is replaced with the OpenVINO Slice operation on the front phase.
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
    Semantic of OpenVINO Slice operation is identical ONNX Slice (opset >= 10).
    It has start, end, steps and axis inputs. It is not in the OpenVINO opset and is replaced to StridedSlice.
    """
    op = 'Slice'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict = None):
        super().__init__(graph, {
            'type': None,
            'op': 'Slice',
            'in_ports_count': 5,
            'out_ports_count': 1,
            'infer': __class__.infer
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
            node.out_port(0).data.set_shape(output_shape)
        else:
            node.out_port(0).data.set_value(input_value[tuple(slice_idx)])


def get_shape_after_slice(input_shape: int, slice_idx: int):
    """
    Calculate shape of a tensor after slicing without actually creating the resulting tensor.
    Is introduced to save memory.
    """
    output_shape = np.zeros(len(input_shape), dtype=np.int32)
    for i, s in enumerate(slice_idx):
        start, end = normalize_slice_indices(input_shape[i], s.start, s.stop)
        output_shape[i] = (end - start) / s.step
    return output_shape


def normalize_slice_indices(size: int, start: int, end: int):
    # converts slice indices to format in which size of slice can be calculated
    start = convert_negative_indices(size, start)
    end = convert_negative_indices(size, end)
    start = clip_indices(size, start)
    end = clip_indices(size, end)
    return start, end


def convert_negative_indices(size: int, val: int):
    """
    Converts negative indices of a tensors: e.g. if val == -1 then convert it to size - 1.
    Note: returned value is not always positive and it's expected behaviour.
    """
    if val < 0:
        return val + size
    else:
        return val


def clip_indices(size: int, val: int):
    # if slice starts and/or ends exceed indices bounds of a tensor this routine cuts them to size or 0
    # Note: returned value is not always positive and it's expected behaviour.
    if val >= size:
        return size
    elif val < 0:
        return 0
    else:
        return val

