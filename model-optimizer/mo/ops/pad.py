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

from mo.graph.graph import Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Pad(Op):
    """ Pad operation that explicitly extends an input tensor at borders.
        
        The operation extends each (not only spatial) dimensions of input tensors by new elements increasing output
        shape.
        The second and third inputs are 1D tensor with number of elements equal to input tensor rank. These inputs
        specify the begin and end paddings.
        The forth input specifies the fill valuu for 'constant' mode and not used for other cases.
    """

    op = 'Pad'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'infer': __class__.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
            'mode': 'constant',
            'fill_value': float(0),
            'force_precision_in_ports': {
                1: 'int64',
                2: 'int64',
            },
        }, attrs)

    def supported_attrs(self):
        return ['mode', 'fill_value', 'pads']

    def backend_attrs(self):
        return [('pad_mode', 'mode'),
                ('pad_value', 'fill_value'),
                ('pads_begin', lambda node: ','.join(map(str, node.pads[:, 0])) if node.has_valid('pads') else None),
                ('pads_end', lambda node: ','.join(map(str, node.pads[:, 1])) if node.has_valid('pads') else None),
                ]

    @staticmethod
    def infer(node):
        pad_node_name = node.soft_get('name', node.id)

        assert len(node.in_nodes()) in [3, 4], "The node {} must have 3 or 4 inputs".format(pad_node_name)

        input_shape = node.in_port(0).data.get_shape()
        pad_beg = node.in_port(1).data.get_value()
        pad_end = node.in_port(2).data.get_value()

        assert pad_beg is not None, 'The padding begin value is None for node {}'.format(pad_node_name)
        assert pad_end is not None, 'The padding end value is None for node {}'.format(pad_node_name)
        assert input_shape is not None, 'The input shape is None for node {}'.format(pad_node_name)
        assert len(input_shape) == len(pad_beg), \
            'Length of begin padding "{}" does not correspond to input tensor shape "{}" for node "{}".' \
            ''.format(pad_beg, input_shape, pad_node_name)
        assert len(input_shape) == len(pad_end), \
            'Length of end padding "{}" does not correspond to input tensor shape "{}" for node "{}".' \
            ''.format(pad_beg, input_shape, pad_node_name)

        node.out_port(0).data.set_shape(input_shape + pad_beg + pad_end)

        if node.in_port(0).data.get_value() is not None:
            pads = np.insert(pad_end, np.arange(len(pad_end)), pad_beg)
            pads = np.reshape(pads, (len(pad_end), 2))
            pad_val = 0
            if len(node.in_nodes()) == 4:
                pad_val = node.in_port(3).data.get_value() if node.in_port(3).data is not None else 0
            node.out_port(0).data.set_value(np.pad(node.in_port(0).data.get_value(), pads, constant_values=pad_val,
                                                   mode='constant'))
        # pad values should be permuted during the NHWC->NCHW layout change
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
        PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')


class AttributedPad(Op):
    """ Pad operation that explicitly extends an input tensor at borders.

        This operation is uses the same semantics as Pad but with pad values specified as attributes.
        Pad values are in format [nDims, 2], where [:, 0] - begin pads, [:, 1] - end pads.
    """

    op = 'AttributedPad'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'infer': None,  # the operation should be replaced before the shape inference
            'in_ports_count': 1,
            'out_ports_count': 1,
            'mode': 'constant',
            'fill_value': float(0),
            'pads': None,
        }, attrs)


class TFPad(Op):
    """ Pad operation that explicitly extends an input tensor at borders.

        This operation with the TensorFlow semantics with inputs:
        1. Input tensor.
        2. Pad values [nDims, 2]
        3. Fill value (Optional)
    """

    op = 'TFPad'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'infer': None,  # the operation should be replaced before the shape inference
            'in_ports_count': 3,
            'out_ports_count': 1,
            'mode': 'constant',
        }, attrs)

class ONNXPad(Op):
    """ Pad operation that explicitly extends an input tensor at borders.

        This operation with the ONNX semantics with inputs:
        1. Input tensor.
        2. Pad values
        3. Fill value (Optional)
    """

    op = 'ONNXPad'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'infer': None,  # the operation should be replaced before the shape inference
            'in_ports_count': 3,
            'out_ports_count': 1,
            'mode': 'constant',
        }, attrs)
