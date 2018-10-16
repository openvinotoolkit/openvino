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
import logging as log

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def tf_slice_infer(node):
    input = node.in_node(0)
    begin = node.in_node(1)
    size = node.in_node(2)
    output = node.out_node()

    if input.value is None or begin.value is None or size.value is None:
        return

    if begin.value.size > 1 or size.value.size > 1:
        log.error("Slice operation doesn't support parameters (begin, size) with size more then 1")
        log.error("  Begin : {}".format(begin.value))
        log.error("  Size  : {}".format(size.value))
        return

    # if the 'size' value is equal to -1 then all remaining elements in dimension are included in the slice.
    # refer to TensorFlow documentation for more details
    if size.value.item() == -1:
        size.value = np.array(input.shape[0] - begin.value.item())
    output.value = input.value[begin.value.item():(begin.value.item() + size.value.item())]
    output.shape = np.array(output.value.shape, dtype=np.int64)


def tf_strided_slice_infer(node):
    begin_id = node.in_node(1).value
    end_id = node.in_node(2).value
    stride = node.in_node(3).value

    shape = node.in_node(0).shape

    if shape is None or any([x < 0 for x in shape]):
        return

    convert_negative_indices(begin_id, shape)
    convert_negative_indices(end_id, shape)

    test_bit = lambda val, offset: ((1 << offset) & val != 0)

    slice_idx = []
    shrink_axis_mask = []
    ellipsis_mask = []
    new_axis_mask = []
    dims = len(begin_id)
    for idx in range(dims):
        l = begin_id[idx] if not test_bit(node.begin_mask, idx) else 0
        r = end_id[idx] if not test_bit(node.end_mask, idx) else shape[idx]

        # Check shrink_axis_mask
        shrink_axis_mask.append(test_bit(node.shrink_axis_mask, idx))
        if shrink_axis_mask[idx]:
            l, r = l, l + 1

        # Check new_axis_mask
        new_axis_mask.append(test_bit(node.new_axis_mask, idx))
        if new_axis_mask[idx]:
            slice_idx.append(np.newaxis)

        # Check ellipsis_mask
        ellipsis_mask.append(test_bit(node.ellipsis_mask, idx))
        if ellipsis_mask[idx]:
            shrink_axis_mask[idx] = False
            l, r = 0, shape[idx]

        slice_idx.append(slice(l, r, stride[idx]))

    value = node.in_node(0).value if node.in_node(0).value is not None else np.zeros(shape)
    value = value[slice_idx]

    for idx, flag in reversed(list(enumerate(shrink_axis_mask))):
        if flag:
            value = np.squeeze(value, idx)

    node['slices'] = np.array(slice_idx)
    node['shrink_axis_mask'] = np.array(shrink_axis_mask)

    node.out_node().value = np.array(value) if node.in_node(0).value is not None else None
    node.out_node().shape = np.array(value.shape)


def convert_negative_indices(indices: np.array, shape: np.array):
    for ind, value in enumerate(indices):
        if value < 0:
            indices[ind] += shape[ind]


def caffe_slice_infer(node):
    """
    Slices an input layer to multiple output layers along a given dimension
    with given slice indices
    Parameters
    ----------
    node

    """
    top_shape = node.in_node(0).shape
    slice_axis = node.axis
    bottom_slice_axis = node.in_node(0).shape[node.axis]
    if len(node.slice_point) == 0:
        new_shape = np.array(top_shape, dtype=np.int64)
        new_shape[slice_axis] = bottom_slice_axis / len(node.out_nodes())
        for i in range(0, len(node.out_nodes())):
            node.out_node(i).shape = new_shape
        return

    assert (len(node.slice_point) == len(node.out_nodes()) - 1)
    prev = 0
    slices = []
    for slice_point in node.slice_point:
        if slice_point <= prev:
            raise Error('Check failed for the layer {}. Slice points should be ordered in increasing manner. '.format(node.id) +
                        'Current slice point {} is not greater than the previous slice point {}. '.format(slice_point, prev) +
                        'Please verify your model correctness')
        slices.append(slice_point - prev)
        prev = slice_point

    slices.append(bottom_slice_axis - prev)
    if sum(slices) != bottom_slice_axis:
        raise Error('Check failed for the layer {}. Sum of slices points {} does not equal '.format(node.id, sum(slices)) +
                    'to the value of input blob shape by the given slice axis {}'.format(bottom_slice_axis))
    for i in range(len(node.out_nodes())):
        new_shape = np.array(top_shape, dtype=np.int64)
        new_shape[slice_axis] = slices[i]
        node.out_node(i).shape = new_shape


def mxnet_slice_axis_infer(node):
    in_shape = node.in_node(0).shape
    slice_axis = node.axis

    new_shape = np.array(in_shape, dtype=np.int64)
    new_shape[slice_axis] = new_shape[slice_axis] / len(node.out_nodes())

    axis_size = in_shape[slice_axis]
    if node.offset < 0:
        node.offset += axis_size

    if not node.dim:
        node.dim = axis_size
    elif node.dim < 0:
        node.dim += axis_size

    input_dim = in_shape.size
    node.dim = (node.dim - node.offset)
    if node.dim > in_shape[slice_axis]:
        raise Error(
            '{0} node dimension value is bigger than the corresponding value in the input shape {1}. ' +
            '\nIn particular {2} is bigger than {3}. The Model Optimizer does not support this case. ' +
            '\nTo overcome, try to edit the original model "end" property of the {0} layer.',
            node.name, ','.join(str(i) for i in in_shape), str(node.dim), str(in_shape[slice_axis])
        )

    for i in range(0, input_dim):
        if i == slice_axis:
            new_shape[i] = node.dim
        else:
            new_shape[i] = in_shape[i]

    for i in range(0, len(node.out_nodes())):
        node.out_node(i)['shape'] = new_shape
