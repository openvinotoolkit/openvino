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

from mo.ops.op import PermuteAttrs
from mo.graph.graph import Node


def part_sizes_to_indices(part_sizes: list):
    """
    Calculates indices of splits in the array based on part sizes for the split.
    Output list can be used as the second argument for np.split function.
    """
    idx = 0
    indices = []
    for part_size in part_sizes:
        idx += part_size
        indices.append(idx)
    # the last element should equal to the size of original array and it is redundant to numpy
    log.debug("part_sizes: {}   -->   indices: {}".format(part_sizes, indices))
    del indices[-1]
    log.debug("part_sizes: {}   -->   indices: {}".format(part_sizes, indices))
    return np.array(indices)


def split(input_data_node: Node, node: Node, axis: int, part_sizes: list):
    """
    Partial inference of generic split node.

    Args:
        @input: input tensor node, subject to split
        @node: node of one of the Split types
        @axis: split dimension index
        @part_sizes: a NumPy array with sizes of all pieces that we split to

    Returns:
        int: normalized axis index

    """

    if input_data_node.shape is None:
        return

    # normalize axis
    if axis < 0:
        axis = input_data_node.shape.size + axis

    if axis < 0 or axis >= input_data_node.shape.size:
        log.error('Model is incorrect: axis for split node is out of range')
        return

    undef_indices = np.argwhere(part_sizes == -1)
    if undef_indices.size > 1:
        log.error('Desired split part sizes have more than one -1 element -- cannot deduce real sizes for them')
        return

    if undef_indices.size == 1:
        undef_index = undef_indices[0]
        part_sizes[undef_index] = 0
        deduced_dim = input_data_node.shape[axis] - np.add.reduce(part_sizes)
        if deduced_dim < 0:
            log.error('Just deduced dimension for the split has negative value that means that split input shape and '
                      'desired parts are not compatible')
            return
        part_sizes[undef_index] = deduced_dim

    all_parts_size = np.add.reduce(part_sizes)
    if all_parts_size != input_data_node.shape[axis]:
        log.error("input.shape[{}] = {}  !=  {} = sum of all parts in part_sizes".format(axis,
                                                                                         input_data_node.shape[axis],
                                                                                         all_parts_size))
        return

    splitted = None
    input_value = input_data_node.value
    if input_value is not None:
        splitted = [np.array(part, dtype=input_value.dtype)
                    for part in np.split(input_value, part_sizes_to_indices(part_sizes), axis)]

    # not all outputs from the split could be used so it is necessary to iterate over output edges and infer shape for
    # necessary nodes only
    for _, dst, edge_attrs in node.graph.out_edges(node.id, data=True):
        out_port = edge_attrs['out']
        out_node = node.out_node(out_port)

        new_out_shape = input_data_node.shape.copy()
        new_out_shape[axis] = part_sizes[out_port]
        node.out_node(out_port).shape = new_out_shape
        if splitted is not None:
            out_node.value = splitted[out_port]
            assert all(out_node.value.shape == out_node.shape)

    node.axis = axis
    # WARNING: != 4 is supposed to work for NHWC to NCHW translation only.
    # if other global permutations happen this will fail
    # TODO: redesign it to have this logic built in NHWC to NCHW translation pass; it requires
    #       additional attributes with layout to be propagated through the network
    if len(input_data_node.shape) != 4 and node.has_valid('dim_attrs') and 'axis' in node.dim_attrs:
        log.warning('Removed "axis" attribute from the scope of the model relayout pass because len(input.shape) == {} '
                    '!= 4 for node {}'.format(len(input_data_node.shape), node.soft_get('name')))
        node.dim_attrs.remove('axis')
        assert 'axis' not in node.dim_attrs
    log.debug('output shapes after split: {}'.format([v.shape for k, v in node.out_nodes().items()]))


def tf_split_infer(node):
    """
    Partial infer of split node similar to Split op of TF.
    """
    # Two inputs: [split_dim, input]
    assert len(node.in_nodes()) == 2, 'Node "{}" must have exactly two inputs'.format(node.soft_get('name'))
    split_dim = node.in_node(0).value
    if split_dim is None:
        log.error('split_dim value for node {} is None. Cannot do shape inference.')
        return

    assert split_dim.ndim == 0, 'The split dimension for node "{}" must be a scalar.'.format(node.soft_get('name'))
    split_dim = split_dim.item()
    input = node.in_node(1)

    if input.shape is None:
        log.error('Input shape for node {} is not defined'.format(node.soft_get('name')))
        return

    log.debug('input shape for split: {}, should be split along {} dim'.format(input.shape, split_dim))
    split_dim_size = input.shape[split_dim]
    log.debug('split_dim_size type = {}'.format(type(split_dim_size)))

    if split_dim_size % node.num_split != 0:
        log.error("split_dim cannot be evenly divided by a given number of parts")
        return

    # split_dim is a numpy array, axis is split_dim[0]
    log.debug('split_dim_size = {}, node.num_split = {}, div = {}, typeof div = {}'.format(
        split_dim_size, node.num_split, split_dim_size / node.num_split, type(split_dim_size / node.num_split)))
    split(input, node, split_dim, [int(split_dim_size / node.num_split)] * node.num_split)
    node.graph.remove_edge(node.in_node(0).id, node.id)
    node['input_port'] = 1

    PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:1')])


def tf_split_v_infer(node: Node):
    """
    Partial infer of split node similar to SplitV op of TF.
    """

    if len(node.in_nodes()) == 1 and not (node.has_valid('axis') and node.has_valid('size_splits')):
        return

    if len(node.in_nodes()) == 3 and (node.has_valid('axis') or node.has_valid('size_splits')):
        return

    # Three inputs: [input, size_splits, split_dim)
    if len(node.in_nodes()) == 3:
        split_dim = node.in_node(2).value
        assert split_dim.ndim == 0
        split_dim = split_dim.item()
        size_splits = node.in_node(1).value
        node.graph.remove_edge(node.in_node(1).id, node.id)
        node.graph.remove_edge(node.in_node(2).id, node.id)
    else:
        split_dim = node.axis
        size_splits = node.size_splits
   
    if split_dim is None:
        log.error('split_dim value for node {} is None. Cannot do shape inference.')
        return
    
    input = node.in_node(0)
    if input.shape is None or size_splits is None:
        log.error('input shape or size of splits are not defined for node {}'.format(node.soft_get('name')))
        return

    node['size_splits'] = size_splits

    log.debug('split_dim = {}, input.shape = {}, size_splits.value = {}'.format(split_dim, input.shape, size_splits))

    # split_dim is a numpy array, axis is split_dim
    split(input, node, split_dim, size_splits)

    PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


def tf_unpack_infer(node: Node):
    if len(node.in_nodes()) != 1:
        log.debug('Unpack node "{}" must have one input.'.format(node.name))
        return

    in_shape = node.in_node().shape
    if in_shape is None:
        log.debug('Unpack node "{}" input node shape is not defined.'.format(node.name))
        return

    split_dim = node.axis
    log.debug('input shape for unpack: {}, should be split along {} dim'.format(in_shape, split_dim))
    split_dim_size = in_shape[split_dim]
    log.debug('split_dim_size type = {}'.format(type(split_dim_size)))

    if node.num_split is not None and node.num_split != split_dim_size:
        log.debug('The unpack where num to unpack is not equal to the size of the dimension to unpack is not supported')
        return

    if node.num_split is None:
        node.num_split = split_dim_size

    if split_dim_size % node.num_split != 0:
        log.error("split_dim cannot be evenly divided by a given number of parts")
        return

    split(node.in_node(), node, split_dim, [int(split_dim_size / node.num_split)] * node.num_split)
    # node shapes will be squeezed in the separate pass
