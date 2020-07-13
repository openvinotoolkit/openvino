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
from mo.utils.shape import get_shape_after_slice

"""
In each framework (or in different opsets of the same framework) Slice operation either has different semantic or 
different parameters, or/and they specified differently in one case as attributes in other case as inputs. To 
distinguish them and not confuse with OpenVINO Slice these operations were added. OpenVINO Slice operation is same as 
ONNX Slice in opset >= 10. To unify all of them before middle phase these replacements take place on the front phase:
    AttributedSlice, TFSlice -> Slice 
    CaffeSlice -> Split 
    MXSlice -> StridedSlice 
"""


class AttributedSlice(Op):
    """
    AttributedSlice is used in old versions of ONNX models (opset version < 10). This operation is
    used only for extracting. On the front phase is replaced with Slice.
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
        return ['axis', 'start', 'end']


class CaffeSlice(Op):
    """
    Slice in Caffe is equivalent to Split operation in OpenVINO.
    https://caffe.berkeleyvision.org/tutorial/layers/slice.html
    After extracting operations on the front phase CaffeSlices are replaced with OpenVINO Splits operation.
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
    On the front phase it is replaced with Slice operation. If size[i] == -1 is replaced to int32_max value for the end.
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

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': 'Slice',
            'in_ports_count': 4,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)

    @staticmethod
    def infer(node: Node):
        input_value = node.in_port(0).data.get_value()
        input_shape = node.in_port(0).data.get_shape()

        start = node.in_port(1).data.get_value()
        end = node.in_port(2).data.get_value()
        if start is None or end is None:
            raise Error('The non-constant start/end values for Slice operation "{}" is not supported'.format(node.name))

        if node.is_in_port_connected(3):
            axis = node.in_port(3).data.get_value()
            if axis is None:
                raise Error('The non-constant axis values for Slice operation "{}" is not supported'.format(node.name))
        else:
            axis = [x for x in range(len(start))]

        if node.is_in_port_connected(4):
            steps = node.in_port(4).data.get_value()
            if steps is None:
                raise Error('The non-constant steps values for Slice operation "{}" is not supported'.format(node.name))
        else:
            steps = np.ones(start.size, dtype=np.int64)

        slice_idx = [slice(0, in_shape, 1) for in_shape in input_shape]
        for i in range(len(axis)):
            # Ranged for output value for specified axis
            slice_idx[axis[i]] = slice(start[i], end[i], steps[i])

        if input_value is None:
            output_shape = get_shape_after_slice(input_shape, slice_idx)
            if np.any(output_shape == 0):
                # todo: add unittest for this case
                raise Error("Output shape ({}) for Slice node {} has zero elements".format(output_shape, node.name))
            node.out_port(0).data.set_shape(output_shape)
        else:
            node.out_port(0).data.set_value(input_value[tuple(slice_idx)])
