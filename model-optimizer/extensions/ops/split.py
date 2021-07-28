# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op, PermuteAttrs


def delete_out_port(idx, node: Node):
    for k in range(idx + 1, node.out_ports_count):
        node.out_port(k).get_connection().set_source(node.out_port(k - 1))
    node.out_ports_count -= 1


class VariadicSplitBase(Op):
    op = None
    enabled = False

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)

        op = node.soft_get('op', None)
        assert op is not None and op in ['VariadicSplit', 'AttributedVariadicSplit'], \
            'Unexpected `op`={} attribute for Split-like node {}'.format(op, name)

        num_in_ports = 1 if op == 'AttributedVariadicSplit' else 3 if op == 'VariadicSplit' else None
        assert num_in_ports in [1, 3], \
            'VariadicSplitBase supports AttributedVariadicSplit with 1 input and VariadicSplit with 3 inputs, ' \
            'but it is {} for {} node {}'.format(num_in_ports, op, name)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == num_in_ports and all([i in connected_inputs for i in range(num_in_ports)]), \
            "{} should have {} connected input ports, but it doesn't for node: `{}`. Ports: {}" \
            "".format(op, num_in_ports, name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None

        axis = node.in_port(1).data.get_value() if op == 'VariadicSplit' else node.soft_get('axis', None)
        assert axis is not None, '{} `axis` is unknown for node {}'.format(op, name)
        assert axis.ndim == 0 or (axis.ndim == 1 and axis.shape[0] == 1), \
            '{} `axis` should be scalar or tensor with shape [1], but it`s not for node {}'.format(op, name)

        split_lengths = node.in_port(2).data.get_value() if op == 'VariadicSplit' else node.soft_get('split_lengths',
                                                                                                     None)
        assert split_lengths is not None, '{} `split_lengths` is unknown for node {}'.format(op, name)

        undefined_elements = np.argwhere(split_lengths == -1).flatten()
        assert undefined_elements.size <= 1, \
            '{} split_lengths=`{}` is a list with output sizes, only one of which could be -1. Node: {}' \
            ''.format(op, split_lengths, name)

        input_elements = input_shape[axis]
        assert undefined_elements.size != 0 or input_elements == np.sum(split_lengths), \
            'The sum of split_lengths=`{}` must match data.shape[axis]=`{}`. Node: {}' \
            ''.format(split_lengths, input_elements, name)

        assert len(split_lengths) >= len([port for i, port in node.out_ports().items() if not port.disconnected()]), \
            'Number of split_lengths=`{}` is less than connected output ports. Node: {}'.format(split_lengths, name)

        # in split_lengths some value can be 0, in this case we will ignore it:
        #     * remove according branch
        #     * remove 0 from split_lengths
        for i in reversed(range(len(split_lengths))):
            if split_lengths[i] == 0:
                if node.out_port(i).disconnected():
                    size_splits = list(split_lengths)
                    split_lengths = np.delete(int64_array(split_lengths), i)
                    if op == 'VariadicSplit':
                        node.in_port(2).data.set_value(split_lengths)
                    else:
                        node['split_lengths'] = split_lengths
                    delete_out_port(i, node)
                else:
                    log.error("Zero dimension on {} branch after Split node {}".format(i, node.id))
                    return

        # shape propagation
        idxs, curr_pos = [], 0
        for i, piece in enumerate(split_lengths):
            assert piece >= -1, 'VariadicSplit split_lengths=`{}` should be non-negative'.format(split_lengths)
            out_shape = input_shape.copy()

            split_length = piece if piece > -1 else input_elements - (np.sum(split_lengths) + 1)
            out_shape[axis] = split_length
            curr_pos = curr_pos + split_length
            idxs.append(curr_pos)

            if not node.out_port(i).disconnected():
                node.out_port(i).data.set_shape(out_shape)

        # value propagation
        input_value = node.in_port(0).data.get_value()
        if input_value is not None:
            split = np.split(input_value, idxs[:-1], axis)
            for i, port in node.out_ports().items():
                if not port.disconnected():
                    port.data.set_value(split[i])

        if op == 'VariadicSplit':
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')
        elif op == 'AttributedVariadicSplit':
            PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


class VariadicSplit(VariadicSplitBase):
    op = 'VariadicSplit'

    def __init__(self, graph: Graph, attrs: dict):
        assert 'axis' not in attrs, \
            'Please use `AttributedVariadicSplit` instead of `VariadicSplit` operation to create node with `axis` ' \
            'parameter set or keep using VariadicSplit operation, but express axis as a scalar second input of ' \
            'VariadicSplit operation'

        assert 'size_splits' not in attrs, \
            'Please use `AttributedVariadicSplit` instead of `VariadicSplit` operation to create node with ' \
            '`size_splits` parameter set or keep using VariadicSplit operation, but express size_splits as a 1D ' \
            'third input of VariadicSplit operation'

        assert 'out_ports_count' in attrs, 'Please set `out_ports_count` attribute for VariadicSplit while creating'

        super().__init__(graph, {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,

            'in_ports_count': 3,
        }, attrs)

    def supported_attrs(self):
        return ['axis']


class AttributedVariadicSplit(VariadicSplitBase):
    op = 'AttributedVariadicSplit'

    def __init__(self, graph: Graph, attrs: dict):
        assert 'axis' in attrs, 'AttributedVariadicSplit operation should have `axis` parameter set while creation'
        assert 'size_splits' in attrs, \
            'AttributedVariadicSplit operation should have `size_splits` parameter set while creation'

        if 'out_ports_count' not in attrs:
            attrs['out_ports_count'] = len(attrs['size_splits'])

        super().__init__(graph, {
            'op': self.op,
            'type': 'VariadicSplit',
            'version': 'opset1',

            'infer': self.infer,

            'in_ports_count': 1,
        }, attrs)


class SplitBase(Op):
    op = None
    enabled = False

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)

        op = node.soft_get('op', None)
        assert op is not None and op in ['Split', 'AttributedSplit'], \
            'Unexpected `op`={} attribute for Split-like node {}'.format(op, name)

        num_in_ports = 1 if op == 'AttributedSplit' else 2 if op == 'Split' else None
        assert num_in_ports in [1, 2], \
            'SplitBase supports AttributedSplit with 1 input and Split with 2 inputs, but it is {} for {} node {}' \
            ''.format(num_in_ports, op, name)

        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == num_in_ports and all([i in connected_inputs for i in range(num_in_ports)]), \
            "{} should have {} connected input ports, but it doesn't for node: `{}`. Ports: {}" \
            "".format(op, num_in_ports, name, connected_inputs)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Input shape is unknown for node {}'.format(name)
        assert node.has_valid('num_splits'), 'Parameter `num_splits` is unknown for node {}'.format(name)
        num_splits = node.num_splits

        axis = node.in_port(1).data.get_value() if op == 'Split' else node.soft_get('axis', None)
        assert axis is not None, '{} `axis` is unknown for node {}'.format(op, name)
        assert axis.ndim == 0, '{} `axis` should be scalar, but it`s not for node {}'.format(op, name)

        assert input_shape[axis] % num_splits == 0, \
            'Input shape is not evenly divided by `num_splits` of {} node {}. `input_shape`={}, `axis`={}, ' \
            '`num_splits`={}'.format(op, name, input_shape, axis, num_splits)

        out_shape = input_shape.copy()
        out_shape[axis] = np.int64(input_shape[axis] / num_splits)

        input_value = node.in_port(0).data.get_value()
        output_value = np.split(input_value.copy(), axis=axis, indices_or_sections=num_splits) \
            if input_value is not None else None

        for idx, port in node.out_ports().items():
            if idx in node.out_nodes():
                port.data.set_shape(out_shape)
                if output_value is not None:
                    port.data.set_value(output_value[idx])

        if op == 'Split':
            PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')
        elif op == 'AttributedSplit':
            PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


class Split(SplitBase):
    op = 'Split'

    def __init__(self, graph: Graph, attrs: dict):
        assert 'num_splits' in attrs, 'Split operation should have `num_splits` while creation'
        if 'out_ports_count' not in attrs:
            attrs['out_ports_count'] = attrs['num_splits']

        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,

            'in_ports_count': 2,
        }, attrs)

        assert 'axis' not in self.attrs, \
            'Please use `AttributedSplit` instead of `Split` operation to create node with `axis` parameter set or' \
            ' keep using Split operation, but express axis as a scalar second input of Split operation'

    def supported_attrs(self):
        return ['num_splits']


class AttributedSplit(SplitBase):
    op = 'AttributedSplit'

    def __init__(self, graph: Graph, attrs: dict):
        assert 'num_splits' in attrs, 'AttributedSplit operation should have `num_splits` while creation'
        if 'out_ports_count' not in attrs:
            attrs['out_ports_count'] = attrs['num_splits']

        super().__init__(graph, {
            'op': self.op,
            'type': 'Split',
            'version': 'opset1',

            'axis': 1,

            'infer': self.infer,

            'in_ports_count': 1,
        }, attrs)

        assert 'axis' in self.attrs, 'AttributedSplit operation should have `axis` parameter set while creation'

    def supported_attrs(self):
        return ['num_splits', 'axis']
