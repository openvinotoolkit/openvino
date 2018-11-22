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

import numpy as np

from mo.front.common.partial_infer.utils import assign_dims_to_weights, int64_array
from mo.front.common.partial_infer.utils import mark_input_bins
from mo.ops.op import PermuteAttrs
from mo.utils.error import Error


def caffe_inner_product(node):
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return
    batches = input_shape[0]
    input_channels = np.prod(input_shape[1:])
    if not node.has_valid('out-size'):
        node['out-size'] = (np.prod(node.in_node(1).shape) / input_channels).astype(np.int64)
    output_channels = node['out-size']

    weights_shape = np.array([output_channels, input_channels], dtype=np.int64)

    # In case if original weight layout is IO we transpose them
    if np.array_equal(node.in_node(1).shape, weights_shape[::-1]) and node.soft_get('transpose_weights') is True:
        node.in_node(1).value = np.transpose(node.in_node(1).value)

    node.out_node().shape = np.array([batches, output_channels], dtype=np.int64)
    # Back propagation of shape to weights
    node.in_node(1).shape = np.array(weights_shape)
    node.in_node(1).value.shape = node.in_node(1).shape

    mark_input_bins(node)
    assign_dims_to_weights(node.in_node(1), None, 1, 0, 2)
    PermuteAttrs.set_permutation(node.in_node(1), node, None)


def onnx_matmul_infer(node):
    if len(node.in_nodes()) != 2:
        raise Error("Wrong number of input nodes for {} node. Should be 2 instead of {}".format(node.name,
                                                                                                len(node.in_nodes())))

    input_node = node.in_node(0)
    weights_node = node.in_node(1)

    input_shape = input_node.shape
    weights_shape = weights_node.shape

    if len(weights_shape) > 2:
        raise Error("MatMul {} with weights shape != 2 is not supported".format(node.name))

    mark_input_bins(node)
    assign_dims_to_weights(weights_node, None, 0, 1, 2)
    PermuteAttrs.set_permutation(weights_node, node, PermuteAttrs.Permutation(perm=int64_array([1, 0]),
                                                                              inv=int64_array([0, 1])))

    node['out-size'] = weights_shape[1]
    node.out_node().shape = np.array([*input_shape[0:-1], weights_shape[1]])
