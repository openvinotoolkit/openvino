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

from mo.front.caffe.extractors.utils import get_canonical_axis_index


def crop_infer(node):
    """
    Crops the shape of the output blob according to input ones be specified params.
    Node should have 2 input blobs - 1st blob is getting cropped by specified axis according
    to the the 2nd (reference) blob.
    The result blob is written to output node shape, and reference blob is removed from graph.
    In order to save the reference dims, it is written to dims parameter.

    Parameters
    ----------
    node


    """
    N = len(node.in_nodes())
    if N < 2:
        log.debug('Wrong number of bottom blobs in ' + node.node)
        return

    shapes = [node.in_node(i).shape for i in range(N)]
    if any(s is None for s in shapes):
        return

    input_shape = np.array(shapes[0])
    start_axis = get_canonical_axis_index(input_shape, node.axis)
    node.axis = start_axis

    reference_shape = np.array(shapes[1])
    input_dim = input_shape.size

    # set new shape to current shape
    new_shape = input_shape.copy()
    ir_axis = []
    ir_offset = []
    dim = []

    for i in range(0, input_dim):
        if i < start_axis:
            new_shape[i] = input_shape[i]
            continue

        crop_offset = 0
        if len(node.offset) == 1:
            crop_offset = node.offset[0]
        elif len(node.offset) > 1:
            crop_offset = node.offset[i - start_axis]

        if input_shape[i] - crop_offset < reference_shape[i]:
            log.error('The crop for dimension is out of bounds in ' + node.node)
            return

        dim.append(reference_shape[i])
        ir_axis.append(i)
        ir_offset.append(crop_offset)
        new_shape[i] = reference_shape[i]

    node.axis = ir_axis
    node.offset = ir_offset
    node.dim = dim
    node.out_node().shape = new_shape
