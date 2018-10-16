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

import logging as log

import numpy as np

from mo.front.caffe.extractors.utils import get_canonical_axis_index


def flatten_infer(node):
    """
    Infers shape of flatten node as it is done in Caffe.
    Output shape: [Batch is the same, Production of other dims]
    Args:
        node: graph flatten node

    """
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return

    # TODO: Should check that input_shape[1:] part doesn't contain -1 elements
    axis = get_canonical_axis_index(input_shape, node.axis)
    end_axis = node.end_axis if node.has('end_axis') else -1
    end_axis = get_canonical_axis_index(input_shape, end_axis)
    prod_axes = np.prod(input_shape[axis: end_axis + 1])
    node.out_node(0).shape = np.array([*input_shape[0: axis], prod_axes, *input_shape[end_axis + 1:]], dtype=np.int64)
    log.debug('input_shape: {}, output_shape: {}'.format(input_shape, node.out_node().shape))

