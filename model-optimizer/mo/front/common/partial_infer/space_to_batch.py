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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array


def space_to_batch_infer(node):
    """
    https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch
    """
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return

    if len(node.in_nodes()) != 3:
        return

    if node.in_node(1).value is None or node.in_node(2).value is None:
        return

    block_size = node.in_node(1).value
    pad = node.in_node(2).value

    pads = pad[:, 0] + input_shape[1:len(block_size)+1] + pad[:, 1]

    node.out_node().shape = int64_array([input_shape[0] * np.prod(block_size),
                                         *[int(x) for x in (pads / block_size)],
                                         *input_shape[len(block_size) + 1:]])


def batch_to_space_infer(node):
    """
    https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch
    """
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return

    if len(node.in_nodes()) != 3:
        return

    if node.in_node(1).value is None or node.in_node(2).value is None:
        return

    block_size = node.in_node(1).value
    crop = node.in_node(2).value

    pads = block_size * input_shape[1:len(block_size)+1]

    sizes = pads - crop[:, 0] - crop[:, 1]
    batch = int(input_shape[0] / (np.prod(block_size)))

    node.out_node().shape = int64_array([batch, *sizes, *input_shape[len(block_size) + 1:]])
