# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

from .special_operations import OPERATIONS_WITH_BIAS


class PatternBuilder:
    def __init__(self):
        self._name_counter = Counter()
        self.pattern = {
            'name': '',
            'nodes': [],
            'edges': []
        }

    def _make_unique_name(self, name):
        self._name_counter.update([name])
        if self._name_counter[name] == 1:
            return name
        return '{}_{}'.format(name, self._name_counter[name] - 1)

    def _insert_atomic_subpattern(self, subpattern, input_nodes, output_nodes, remove_orig_edge=True):
        if remove_orig_edge and input_nodes and output_nodes:
            if not isinstance(input_nodes, list):
                input_nodes = [input_nodes]
            if not isinstance(output_nodes, list):
                output_nodes = [output_nodes]
            for input_node in input_nodes:
                for output_node in output_nodes:
                    if (input_node, output_node) in self.pattern['edges']:
                        self.pattern['edges'].remove((input_node, output_node))

        for key in ['nodes', 'edges']:
            self.pattern[key] += subpattern[key]
        return self

    def append_op_const(self, op_type, op_name=None, remove_orig_edge=True):
        return self.insert_op_const(None, None, op_type, op_name, remove_orig_edge)

    def insert_op_const(self, input_node, output_node, op_type, op_name=None, remove_orig_edge=True):
        if not op_name:
            op_name = op_type
        op_name = op_name.lower()
        op_name_const = op_name + '_const'
        op_name_const_data = op_name + '_const_data'
        op_name_data = op_name + '_data'
        new_names = [self._make_unique_name(name) for name in
                     [op_name, op_name_const, op_name_const_data, op_name_data]]
        op_name, op_name_const, op_name_const_data, op_name_data = new_names

        op_pattern = dict(
            nodes=[
                (op_name_const, {'kind': 'op'}),
                (op_name_const_data, {'kind': 'data', 'value': lambda v: v is not None}),
                (op_name, {'kind': 'op', 'type': op_type}),
                (op_name_data, {'kind': 'data'})
            ],
            edges=[
                (op_name_const, op_name_const_data),
                (op_name_const_data, op_name),
                (op_name, op_name_data),
            ]
        )
        input_node = self._use_tail_as_input(input_node, output_node)
        if input_node:
            op_pattern['edges'].insert(0, (input_node, op_name))
        if output_node:
            op_pattern['edges'].append((op_name_data, output_node))
        return self._insert_atomic_subpattern(op_pattern, input_node, output_node, remove_orig_edge)

    def insert_multiply_const(self, input_node=None, output_node=None, op_name=None, remove_orig_edge=True):
        return self.insert_op_const(input_node, output_node, 'Multiply', op_name, remove_orig_edge)

    def insert_add_const(self, input_node=None, output_node=None, op_name=None, remove_orig_edge=True):
        return self.insert_op_const(input_node, output_node, 'Add', op_name, remove_orig_edge)

    def insert_bias(self, input_node=None, output_node=None, remove_orig_edge=True):
        return self.insert_op_const(input_node, output_node, 'Add', 'bias', remove_orig_edge)

    def insert_scaleshift(self, input_node=None, output_node=None, remove_orig_edge=True):
        self.insert_multiply_const(input_node, output_node, 'scaleshift_multiply', remove_orig_edge)
        return self.insert_add_const(self._tail(), output_node, 'scaleshift_add')

    def insert_split(self, input_node=None, output_node=None, op_name=None, remove_orig_edge=True):
        return self.insert_op_const(input_node, output_node, 'Split', op_name, remove_orig_edge)

    def insert_single_op(self, input_nodes, output_nodes, op_type, op_name, remove_orig_edge=True):
        name_data = self._make_unique_name(op_name + '_data')
        op_name = self._make_unique_name(op_name)
        op_pattern = dict(
            nodes=[
                (op_name, {'kind': 'op', 'type': op_type})
            ],
            edges=[]
        )
        if op_type != 'Result':
            op_pattern['nodes'].append((name_data, {'kind': 'data'}))
            op_pattern['edges'].append((op_name, name_data))
        input_nodes = self._use_tail_as_input(input_nodes, output_nodes)
        if input_nodes:
            if not isinstance(input_nodes, list):
                input_nodes = [input_nodes]
            for i, input_node in enumerate(input_nodes):
                op_pattern['edges'].insert(i, (input_node, op_name))

        if output_nodes:
            if not isinstance(output_nodes, list):
                output_nodes = [output_nodes]
            for output_node in output_nodes:
                op_pattern['edges'].append((name_data, output_node))

        return self._insert_atomic_subpattern(op_pattern, input_nodes, output_nodes, remove_orig_edge)

    def append_single_op(self, op_type, op_name, remove_orig_edge=True):
        return self.insert_single_op(None, None, op_type, op_name, remove_orig_edge)

    def _tail(self):
        return self.pattern['edges'][-1][1]

    def _use_tail_as_input(self, input_nodes, output_nodes):
        # if we have edges in pattern than the first node already has inserted earlier
        if self.pattern['edges'] and not output_nodes:
            if not input_nodes:
                return self._tail()
            if isinstance(input_nodes, list):
                return [node if node else self._tail() for node in input_nodes]
        return input_nodes

    def insert_activation(self, input_node=None, output_node=None, remove_orig_edge=True):
        return self.insert_single_op(
            input_node,
            output_node,
            lambda x: x in ['ReLU', 'PReLU', 'Activation', 'Sigmoid'],
            'activation',
            remove_orig_edge
        )

    def insert_conv_fc(self, input_node=None, output_node=None, name='conv_fc', remove_orig_edge=True):
        return self.insert_single_op(
            input_node,
            output_node,
            lambda x: x in [op['type'] for op in OPERATIONS_WITH_BIAS],
            name,
            remove_orig_edge
        )

    def insert_se(self, input_node=None, output_node=None,
                  start_name='se_start', end_name='se_end', remove_orig_edge=True, is_swish=False):
        self.insert_single_op(input_node, output_node, 'ReduceMean', start_name, remove_orig_edge)
        self.insert_conv_fc(output_node=output_node)
        self.insert_bias(output_node=output_node)
        if is_swish:
            self.insert_swish(output_node=output_node)
        else:
            self.insert_single_op(None, output_node, lambda x: x in ['ReLU', 'PReLU'], 'act')
        self.insert_conv_fc(output_node=output_node)
        self.insert_bias(output_node=output_node)
        self.insert_single_op(None, output_node, 'Sigmoid', 'act')
        return self.insert_single_op(None, output_node, 'Multiply', end_name)

    def insert_swish(self, input_node=None, output_node=None, remove_orig_edge=True):
        if not input_node:
            input_node = self._tail()
        return self.insert_single_op(input_node, output_node, 'Swish', 'swish', remove_orig_edge)

    def get_last_node(self):
        return self.pattern['nodes'][-1][0]

    def set_name(self, pattern_name):
        self.pattern['name'] = pattern_name
        return self
