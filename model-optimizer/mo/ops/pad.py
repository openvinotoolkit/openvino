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

from mo.graph.graph import Graph
from mo.ops.op import Op, PermuteAttrs


class Pad(Op):
    """ Pad operation that explicitly extends an input tensor at edges.
        
        This operation frequently appears in TF and rarely in ONNX models
        followed by some windowed operation like convolution or pooling.
        The operation extends each (not only spatial) dimensions of input
        tensors by new elements increasing output shape. The filling values
        is defined by 'mode' and 'fill_value' attributes, but usually it is zero
        padding.

        The operation has two forms: with one or two input arguments.
        The first aruments is an input tensor to be padded. The second
        argument is an optional padding values of shape Nx2, where N is
        a number of dimensions in an input tensor:

            [[pad_begin_dim1, pad_end_dim1],
             [pad_begin_dim2, pad_end_dim2],
             ...
             [pad_begin_dimN, pad_end_dimN]]

        where pad_begin_dim1 etc. are padding margins in elements. If the second
        input argument is omitted, then it is in 'pads' attribute in the same
        format.
    """

    op = 'Pad'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'mode': 'constant',
            'fill_value': float(0),
            'pads': None
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
        PermuteAttrs.create_permute_attrs(node, attrs=[('pads', 'input:0')])

        if node.has_valid('pads'):
            assert len(node.in_nodes()) == 1, "Pad operation has pads attribute and unexpected additional input " \
                                              "argument for node {}.".format(node.name)
        else:
            assert len(node.in_nodes()) >= 2, "Missing required second input argument for node {} and pads attribute " \
                                              "is missing.".format(node.name)
            node.pads = node.in_node(1).value
            if len(node.in_nodes()) == 3:  # the third input contains the fill value
                node.fill_value = node.in_node(2).value
        padding = node.pads

        input_shape = node.in_node(0).shape
        if padding is None or input_shape is None:
            log.error('The paddings are not defined for node "{}"'.format(node.soft_get('name')))
            return

        # paddings can be defined, partially defined or undefined
        # TODO for now we only handle fully defined paddings
        # That means that intermediate tensor that delivers padding
        # should have defined value and size Nx2
        # TODO possible broadcasts are not supported
        assert (padding.ndim == 2 and padding.shape[1] == 2)

        # make sure that input has the same number of dimensions as the number of padding dimensions
        assert (padding.shape[0] == len(input_shape)), \
            "Input tensor shape {} and pads values {} do not match for Pad node {}".format(
                input_shape, padding.shape, node.name
            )

        # sum low and high padding values to calculate the shape modification vector
        shape_change = np.add.reduce(padding, 1)
        assert (shape_change.shape == input_shape.shape)

        # preserve non-positive values in the input shape, because it has a special meaning
        shape = np.array(
            [shape_change[i] + input_shape[i] if input_shape[i] > 0 else input_shape[i] for i in
             range(len(input_shape))])

        assert len(node.out_nodes()) == 1

        node.out_node().shape = shape
