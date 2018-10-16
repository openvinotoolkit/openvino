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


def tf_reduce_infer(node, op=None):
    input_shape = node.in_node(0).shape
    log.debug("input_shape: {}".format(input_shape))
    axis = node.in_node(1).value
    if input_shape is None or axis is None or input_shape.ndim != 1 or axis.ndim > 1:
        return
    output_shape = np.array(input_shape)
    if node.keep_dims:
        output_shape[axis] = 1
    else:
        output_shape = np.delete(output_shape, axis)
    node.out_node().shape = output_shape
    if op is not None and node.in_node(0).value is not None:
        node.out_node(0).value = np.array([op(node.in_node(0).value, (*axis,))],
                                          dtype=node.in_node(0).value.dtype)  # TODO extend to multi-dimensional axis
        log.debug("value: {}".format(node.out_node(0).value))
