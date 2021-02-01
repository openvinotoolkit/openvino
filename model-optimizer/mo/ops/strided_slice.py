"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error
from mo.utils.utils import array_to_str


class StridedSlice(Op):
    op = 'StridedSlice'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': 'StridedSlice',
            'version': 'opset1',
            'in_ports_count': 4,
            'out_ports_count': 1,
            'infer': __class__.infer
        }, attrs)
        assert 'new_axis_mask' in attrs, "Attribute 'new_axis_mask' of the StridedSlice node is not given."
        assert 'shrink_axis_mask' in attrs, "Attribute 'shrink_axis_mask' of the StridedSlice node is not given."
        assert 'ellipsis_mask' in attrs, "Attribute 'ellipsis_mask' of the StridedSlice node is not given."
        assert 'begin_mask' in attrs, "Attribute 'begin_mask' of the StridedSlice node is not given."
        assert 'end_mask' in attrs, "Attribute 'end_mask' of the StridedSlice node is not given."

    def backend_attrs(self):
        al = list()

        def convert(attr):
            return lambda node: array_to_str(node, attr)

        for a in list(['new_axis_mask', 'shrink_axis_mask', 'ellipsis_mask', 'begin_mask', 'end_mask']):
            al.append((a, convert(a)))
        return al

    @staticmethod
    def infer(node: Node):
        # FW assures that begin, end, and strides are of the same length th
        tf_strided_slice_infer(node)

        out_shape = node.out_port(0).data.get_shape()
        assert out_shape is not None, \
            'Output shape was not calculated for node {}'.format(node.name)

        PermuteAttrs.create_permute_attrs(node, attrs=[('shrink_axis_mask', 'input:0', permute_masks),
                                                       ('new_axis_mask', 'input:0', permute_masks),
                                                       ('ellipsis_mask', 'input:0', permute_masks),
                                                       ('begin_mask', 'input:0', permute_masks),
                                                       ('end_mask', 'input:0', permute_masks)])

        # PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')
        # PermuteInputs().set_input_permutation(node.in_node(2), node, 'input:0', 'shape')
        # PermuteInputs().set_input_permutation(node.in_node(3), node, 'input:0', 'shape')


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

    new_axis_mask = extend_mask(node.new_axis_mask, dims)
    shrink_axis_mask = extend_mask(node.shrink_axis_mask, dims)
    ellipsis_mask = extend_mask(node.ellipsis_mask, dims)
    begin_mask = extend_mask(node.begin_mask, dims)
    end_mask = extend_mask(node.end_mask, dims)

    old_idx = 0
    ellips_ext = 0
    id_em = 0
    for idx in range(dims):
        if new_axis_mask[idx]:
            slice_idx.append(np.newaxis)
        elif ellipsis_mask[idx]:
            ellips_ext = len(shape) - (dims - np.count_nonzero(new_axis_mask) - 1)
            id_em = idx
            for i in range(0, ellips_ext):
                slice_idx.append(slice(0, shape[old_idx], 1))
                old_idx = old_idx + 1
        else:
            s = stride[idx] if len(stride) > idx else 1
            def_beg = 0 if s > 0 else -1
            def_end = shape[old_idx] if s > 0 else -shape[old_idx]-1
            l = begin_id[idx] if begin_mask[idx] and idx < len(begin_id) else def_beg
            r = end_id[idx] if end_mask[idx] and idx < len(end_id) else def_end

            # Check shrink_axis_mask
            if shrink_axis_mask[idx] and idx < len(shape):
                slice_idx.append(slice(l, l+1, s))
            else:
                slice_idx.append(slice(l, r, s))
            old_idx = old_idx + 1

    value = node.in_node(0).value if node.in_node(0).value is not None else np.zeros(shape)
    value = value[tuple(slice_idx)]

    for idx, flag in reversed(list(enumerate(shrink_axis_mask))):
        if flag:
            if ellips_ext > 0 and idx > id_em:
                idx = idx + ellips_ext - 1
            try:
                value = np.squeeze(value, idx)
            except ValueError:
                # ignore this error
                continue

    for i, s in enumerate(slice_idx):
        if s is None:
            slice_idx[i] = slice(0, 1, 1)

    node['slices'] = np.array(slice_idx)
    for attr in ('shrink_axis_mask', 'new_axis_mask', 'ellipsis_mask', 'begin_mask', 'end_mask'):
        node[attr] = np.array(node[attr], dtype=np.int32)

    node['force_precision_in_ports'] = {port: 'int64' for port in range(1, len(node.in_nodes()))}

    node.out_node().value = value.copy() if node.in_node(0).value is not None else None
    node.out_node().shape = np.array(value.shape, dtype=np.int64)


def convert_negative_indices(indices: np.array, shape: np.array):
    for ind, value in enumerate(indices):
        if value < 0:
            indices[ind] += shape[ind]

def permute_array(node: Node, array: np.array):
    """
    This function permutes masks according to permutation parameter. Mask have the same or more length than output
    """
    attr_mask_extended = list(array)

    # If input and output have length of shape 3 and less, no need to permute
    if len(node.in_port(0).data.get_shape()) < 4 and len(node.out_port(0).data.get_shape()) < 4:
        return attr_mask_extended

    perm_len = len(node.out_port(0).data.get_shape()) + np.count_nonzero(node.shrink_axis_mask)
    perm_len = len(array)

    perm = PermuteAttrs.get_nhwc_to_nchw_permutation(perm_len)
    perm_list = list(perm.perm)
    # if mask length is more than output, just add tail that will not be permuted to avoid error
    for i in range(perm_len, len(attr_mask_extended)):
        perm_list.append(i)
    return int64_array(attr_mask_extended)[int64_array(perm_list)]


def permute_masks(node: Node, permutation: PermuteAttrs.Permutation, attr: str):
    if not node.has_valid(attr):
        return None

    node[attr] = permute_array(node, node[attr])
    return node[attr]

