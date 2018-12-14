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

from mo.ops.op import PermuteAttrs
from mo.utils.error import Error


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


def split(input, node, outputs, axis, part_sizes):
    """
    Partial inference of generic split node.

    Args:
        @input: input tensor node, subject to split
        @outputs: output tensor nodes where we put inferred output shapes
        @axis: split dimension index
        @part_sizes: a NumPy array with sizes of all pieces that we split to

    Returns:
        int: normalized axis index

    """

    if input.shape is None:
        return

    if len(outputs) != len(part_sizes):
        log.error('Number of outputs do not match the number of parts with sizes.')
        return

    # normalize axis
    if axis < 0:
        axis = input.shape.size + axis

    if axis < 0 or axis >= input.shape.size:
        log.error('Model is incorrect: axis for split node is out of range')
        return

    undef_indices = np.argwhere(part_sizes == -1)
    if undef_indices.size > 1:
        log.error('Desired split part sizes have more than one -1 element -- cannot deduce real sizes for them')
        return

    if undef_indices.size == 1:
        undef_index = undef_indices[0]
        part_sizes[undef_index] = 0
        deduced_dim = input.shape[axis] - np.add.reduce(part_sizes)
        if deduced_dim < 0:
            log.error(
                'Just deduced dimension for the split has negative value that means that split input shape and desired parts are not compatible')
            return

    all_parts_size = np.add.reduce(part_sizes)
    if all_parts_size != input.shape[axis]:
        log.error("input.shape[{}] = {}  !=  {} = sum of all parts in part_sizes".format(axis, input.shape[axis],
                                                                                         all_parts_size))
        return

    for i, part_size in enumerate(part_sizes):
        shape = input.shape.copy()
        shape[axis] = part_size
        outputs[i].shape = shape

    if input.value is not None:
        splitted = np.split(input.value, part_sizes_to_indices(part_sizes), axis)
        # log.debug("splitted = {}".format(splitted))
        for i, part in enumerate(splitted):
            outputs[i].value = part
            # log.debug('outputs[i].value.shape = {}, outputs[i].shape = {}'.format(outputs[i].value.shape, outputs[i].shape))
            assert all(outputs[i].value.shape == outputs[i].shape)

    assert not node.has_valid('axis') or node.axis == axis
    node.axis = axis
    # WARNING: != 4 is supposed to work for NHWC to NCHW translation only; if other global permutations happen this will fail
    # TODO: redesign it to have this logic built in NHWC to NCHW translation pass; it requires
    #       additional attributes with layout to be propagated through the network
    if len(input.shape) != 4 and node.has_valid('dim_attrs') and 'axis' in node.dim_attrs:
        log.warning(
            'Removed "axis" attribute from the scope of the model relayout pass because len(input.shape) == {} != 4 for node {}'.format(
                len(input.shape),
                node.name if node.has_valid('name') else '<UNKNOWN>'))
        node.dim_attrs.remove('axis')
        assert 'axis' not in node.dim_attrs


def tf_split_infer(node):
    """
    Partial infer of split node similar to Split op of TF.
    """

    if len(node.in_nodes()) == 1:
        return True

    # Two inputs: [split_dim, input)
    assert (len(node.in_nodes()) == 2)
    split_dim = node.in_node(0).value
    if split_dim is None:
        log.error('split_dim value for node {} is None. Cannot do shape inference.')
        return
    assert split_dim.ndim == 0
    split_dim = split_dim.item()
    input = node.in_node(1)

    if split_dim is None or input.shape is None:
        return

    log.debug('input shape for split: {}, should be split along {} dim'.format(input.shape, split_dim))
    split_dim_size = input.shape[split_dim]
    log.debug('split_dim_size type = {}'.format(type(split_dim_size)))

    if split_dim_size % node.num_split != 0:
        log.error("split_dim cannot be evenly divided by a given number of parts")
        return

    outputs = node.out_nodes()
    # split_dim is a numpy array, axis is split_dim[0]
    log.debug(
        'split_dim_size = {}, node.num_split = {}, div = {}, typeof div = {}'.format(split_dim_size, node.num_split,
                                                                                     split_dim_size / node.num_split,
                                                                                     type(
                                                                                         split_dim_size / node.num_split)))
    split(input, node, [outputs[i] for i in range(len(outputs))], split_dim,
          [int(split_dim_size / node.num_split)] * node.num_split)
    log.debug('output shapes after split: {}'.format([v.shape for k, v in outputs.items()]))
    node.graph.remove_edge(node.in_node(0).id, node.id)
    node['input_port'] = 1

    PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:1')])


def tf_split_v_infer(node):
    """
    Partial infer of split node similar to SplitV op of TF.
    """

    if len(node.in_nodes()) == 1 and not (node.has_valid('axis') and node.has_valid('size_splits')):
        return True

    if len(node.in_nodes()) == 3 and (node.has_valid('axis') or node.has_valid('size_splits')):
        return True

    # Three inputs: [input, size_splits, split_dim)
    if len(node.in_nodes())==3 :
        split_dim = node.in_node(2).value
        assert split_dim.ndim == 0
        split_dim = split_dim.item()
        size_splits = node.in_node(1).value
        node.graph.remove_edge(node.in_node(1).id, node.id)
        node.graph.remove_edge(node.in_node(2).id, node.id)
    else :
        split_dim = node.axis
        size_splits = node.size_splits
   
    if split_dim is None:
        log.error('split_dim value for node {} is None. Cannot do shape inference.')
        return
    
    input = node.in_node(0)
    
    log.debug(
        'split_dim = {}, input.shape = {}, size_splits.value = {}'.format(split_dim, input.shape, size_splits))

    if split_dim is None or input.shape is None or size_splits is None:
        return

    outputs = node.out_nodes()
    # split_dim is a numpy array, axis is split_dim
    split(input, node, [outputs[i] for i in range(len(outputs))], split_dim, size_splits)
    log.debug('output shapes after split: {}'.format([v.shape for k, v in outputs.items()]))
    
    PermuteAttrs.create_permute_attrs(node, attrs=[('axis','input:0')])

def tf_unpack_infer(node):
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

    outputs = node.out_nodes()
    split(node.in_node(), node, [outputs[i] for i in range(len(outputs))], split_dim,
          [int(split_dim_size / node.num_split)] * node.num_split)

    # node shapes will be squeezed in the separate pass
    log.debug('output shapes after split: {}'.format([v.shape for k, v in outputs.items()]))
