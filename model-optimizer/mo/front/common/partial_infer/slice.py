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

from mo.utils.error import Error


def tf_strided_slice_infer(node):
    if node.in_node(1).value is None or node.in_node(2).value is None:
        raise Error('Strided slice layer supports only constant begin and end inputs')
    begin_id = node.in_node(1).value.copy()
    end_id = node.in_node(2).value.copy()
    if len(node.in_nodes()) > 3:
        if node.in_node(3).value is None:
            raise Error('Strided slice layer supports only constant stride input')
        stride = node.in_node(3).value
    else:
        stride = []

    shape = node.in_node(0).shape

    if shape is None or any([x < 0 for x in shape]):
        return

    convert_negative_indices(begin_id, shape)
    convert_negative_indices(end_id, shape)

    slice_idx = []
    dims = np.amax(np.array([len(begin_id), len(end_id), len(stride),
                             len(node.shrink_axis_mask), len(node.new_axis_mask), len(node.ellipsis_mask),
                             len(node.begin_mask), len(node.end_mask)]))

    # make mask correct length
    def extend_mask(in_mask, fin_len, zeros=True):
        mask = list(in_mask)
        if len(mask) < fin_len:
            if zeros:
                mask.extend(np.zeros(dims-len(mask), dtype=np.int32))
            else:
                mask.extend(np.ones(dims-len(mask), dtype=np.int32))
        return np.array(mask, dtype=np.int32)

    for mask in {'new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask'}:
        node[mask] = extend_mask(node[mask], dims)
    node.begin_mask = extend_mask(node.begin_mask, dims, False)
    node.end_mask = extend_mask(node.end_mask, dims, False)

    old_idx = 0
    ellips_ext = 0
    id_em = 0
    for idx in range(dims):
        if node.new_axis_mask[idx]:
            slice_idx.append(np.newaxis)
        elif node.ellipsis_mask[idx]:
            ellips_ext = len(shape) - (dims - np.count_nonzero(node.new_axis_mask) - 1)
            id_em = idx
            for i in range(0, ellips_ext):
                slice_idx.append(slice(0, shape[old_idx], 1))
                old_idx = old_idx + 1
        else:
            s = stride[idx] if len(stride) > idx else 1
            def_beg = 0 if s > 0 else -1
            def_end = shape[old_idx] if s > 0 else -shape[old_idx]-1
            l = begin_id[idx] if node.begin_mask[idx] and idx < len(begin_id) else def_beg
            r = end_id[idx] if node.end_mask[idx] and idx < len(end_id) else def_end

            # Check shrink_axis_mask
            if node.shrink_axis_mask[idx] and idx < len(shape):
                slice_idx.append(slice(l, l+1, s))
            else:
                slice_idx.append(slice(l, r, s))
            old_idx = old_idx + 1

    value = node.in_node(0).value if node.in_node(0).value is not None else np.zeros(shape)
    # fix for the warning: "FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated use
    # `arr[tuple(seq)]` instead of `arr[seq]`"
    value = value[tuple(slice_idx)]

    for idx, flag in reversed(list(enumerate(node.shrink_axis_mask))):
        if flag:
            if ellips_ext > 0 and idx > id_em:
                idx = idx + ellips_ext - 1
            try:
                value = np.squeeze(value, idx)
            except ValueError:
                # ignore this error
                continue

    node['slices'] = np.array(slice_idx)
    for attr in ('shrink_axis_mask', 'new_axis_mask', 'ellipsis_mask', 'begin_mask', 'end_mask'):
        node[attr] = np.array(node[attr], dtype=np.int32)

    node.out_node().value = np.array(value) if node.in_node(0).value is not None else None
    node.out_node().shape = np.array(value.shape, dtype=np.int64)

    # change precision to I32 for begin, end, stride inputs
    for i in range(1, len(node.in_nodes())):
        inp = node.in_node(i)
        inp["force_precision"] = "I32"


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
            raise Error(
                'Check failed for the layer {}. Slice points should be ordered in increasing manner. '.format(node.id) +
                'Current slice point {} is not greater than the previous slice point {}. '.format(slice_point, prev) +
                'Please verify your model correctness')
        slices.append(slice_point - prev)
        prev = slice_point

    slices.append(bottom_slice_axis - prev)
    if sum(slices) != bottom_slice_axis:
        raise Error(
            'Check failed for the layer {}. Sum of slices points {} does not equal '.format(node.id, sum(slices)) +
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
