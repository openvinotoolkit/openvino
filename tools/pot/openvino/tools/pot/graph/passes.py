# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from copy import copy
from functools import partial
from itertools import cycle
from typing import Dict
from typing import List, Set

import numpy as np
from openvino.tools.mo.back.ForceStrictPrecision import ForceStrictPrecision
from openvino.tools.mo.back.compress_quantized_weights import CompressQuantizeWeights, ZeroPointOptimizer
from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.fakequantize import FakeQuantize
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph, Node, rename_node
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.pattern_match import apply_pattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.middle.passes.convert_data_type import convert_blob

from . import editor as ge
from . import node_utils as nu
from .editor import get_nodes_by_type
from .pattern_utils import get_fq_result_pattern
from .special_operations import OPERATIONS_WITH_WEIGHTS, DETECTION_OUTPUT_FINAL_TYPES, \
    SPLIT_OPERATIONS, OPERATIONS_WITH_BIAS
from .utils import find_operation_matches, is_ignored, get_hw_aware_ignored_patterns
from ..graph.node_utils import get_all_node_outputs, get_node_inputs, get_node_input, get_weights_for_node
from ..graph.special_patterns import get_ignored_patterns
from ..utils.logger import get_logger
from .utils import get_hardware_config_operation_type

logger = get_logger(__name__)

#pylint: disable=C0302
class SaveBNStatistics(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('const_1', dict(op='Const')),
                ('const_2', dict(op='Const')),
                ('const_3', dict(op='Const')),
                ('const_4', dict(op='Const')),
                ('bn', dict(op=lambda x: x in ['FusedBatchNorm',
                                               'BatchNorm',
                                               'BatchNormalization'])),
            ],
            edges=[
                ('input', 'bn', {'in': 0}),
                ('const_1', 'bn', {'in': 1}),
                ('const_2', 'bn', {'in': 2}),
                ('const_3', 'bn', {'in': 3}),
                ('const_4', 'bn', {'in': 4}),
            ])

    def replace_sub_graph(self, _, match):
        input_node = match['input']

        bn = match['bn']
        const_1 = match['const_1']
        const_2 = match['const_2']
        const_3 = match['const_3']
        const_4 = match['const_4']

        input_node['bn_weights'] = {
            'std': const_1.value,
            'mean': const_2.value,
            'input_std': const_3.value,
            'input_mean': const_4.value
        }

        logger.debug('Save BN %s weights to %s node', bn.name, input_node.name)


class InsertFakeQuantize(BackReplacementPattern):

    enabled = False

    @property
    def quantize_operations(self):
        return getattr(self, '_quantize_operations', [])

    @quantize_operations.setter
    def quantize_operations(self, value):
        setattr(self, '_quantize_operations', value)

    @property
    def quantize_output_operations(self):
        return getattr(self, '_quantize_output_operations', [])

    @quantize_output_operations.setter
    def quantize_output_operations(self, value):
        setattr(self, '_quantize_output_operations', value)

    @property
    def hardware_config(self):
        return getattr(self, '_hardware_config', [])

    @hardware_config.setter
    def hardware_config(self, value):
        setattr(self, '_hardware_config', value)

    @property
    def ignored_params(self):
        return getattr(self, '_ignored_params', {'skip_model': False, 'scope': [], 'operations': []})

    @ignored_params.setter
    def ignored_params(self, value):
        setattr(self, '_ignored_params', value)

    def pattern(self):
        op_types = []
        for op in self.quantize_operations:
            op_types.append(op['type'])

        return dict(
            nodes=[
                ('m_op', {'type': lambda x: x in op_types})
            ],
            edges=[]
        )

    @staticmethod
    def quantize_only_input(node: Node):
        if node.type in ['Interpolate', 'Power', 'ReduceMean', 'NormalizeL2',
                         'Assign', 'PReLU', 'ReLU', 'Sigmoid', 'Tanh', 'Clamp', 'MVN']:
            return True
        # ScaleSift case, FQ only for input
        if node.type == 'Multiply' and nu.check_input_data_is_const(node, 1):
            output_node = nu.get_node_output(node, 0)[0]
            if output_node.type == 'Add' and nu.check_input_data_is_const(output_node, 1):
                logger.debug('Scaleshift found at {}->{}'.format(node.name, output_node.name))
                return True
        return False

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        m_op = match['m_op']
        if not find_operation_matches(self.quantize_operations, m_op) \
                or is_ignored(self.ignored_params, m_op):
            return

        if m_op.type in ['Convolution', 'ConvolutionBackpropData', 'MatMul']:
            insert_fake_quantize(graph, m_op, [0, 1], hw_config=self.hardware_config, input_priority_types=self.input_priority_types)
        elif m_op.type == 'LSTMCell':
            insert_fake_quantize(graph, m_op, [0, 1, 2, 3, 4], hw_config=self.hardware_config, input_priority_types=self.input_priority_types)
        elif self.quantize_only_input(m_op):
            insert_fake_quantize(graph, m_op, [0], hw_config=self.hardware_config, input_priority_types=self.input_priority_types)
        else:
            insert_fake_quantize(graph, m_op, hw_config=self.hardware_config, input_priority_types=self.input_priority_types)

        biased_op = [op['type'] for op in OPERATIONS_WITH_BIAS]
        if m_op.type in self.quantize_output_operations:
            bias_node = nu.get_bias_for_node(m_op)
            if m_op.type in biased_op and bias_node:
                m_op = nu.get_node_output(bias_node, 0)[0]
            insert_output_fake_quantize(graph, m_op, hw_config=self.hardware_config,
                                        ignored_params=self.ignored_params)


class FakeQuantizePropagation(BackReplacementPattern):

    enabled = False

    @property
    def hardware_config(self):
        return getattr(self, '_hardware_config', [])

    @hardware_config.setter
    def hardware_config(self, value):
        setattr(self, '_hardware_config', value)


    def remove_node_and_reset_connections(self, graph, node: Node, in_port):
        node.in_port(0).disconnect()
        node.out_port(0).get_connection().set_source(in_port)
        graph.remove_node(node.id)

    def jump_to_first_input(self, graph, fq: Node) -> []:
        in_port = fq.in_port(0).get_source()
        op = in_port.node

        # Disconnect FQ from input and reconnect outputs to input node
        self.remove_node_and_reset_connections(graph, fq, in_port)

        return insert_fake_quantize(graph, op, [0], hw_config=self.hardware_config)

    def jump_to_all_inputs(self, graph: Graph, fq: Node) -> []:
        in_port = fq.in_port(0).get_source()
        op = in_port.node

        # Disconnect FQ from input and reconnect outputs to input node
        self.remove_node_and_reset_connections(graph, fq, in_port)

        # Insert FQ operations for all inputs
        return insert_fake_quantize(graph, op, hw_config=self.hardware_config)

    def jump_to_all_branch_except_const(self, graph, fq: Node) -> []:
        in_port = fq.in_port(0).get_source()
        op = in_port.node
        ports = [i for i in range(op.in_ports_count) if op.in_port(i).get_source() is not None and \
                 op.in_port(i).get_source().node.type != 'Const']

        # Disconnect FQ from input and reconnect outputs to input node
        self.remove_node_and_reset_connections(graph, fq, in_port)

        return insert_fake_quantize(graph, op, ports, hw_config=self.hardware_config)

    def jump_over_split_concat(self, graph: Graph, fq: Node) -> []:
        in_port = fq.in_port(0).get_source()
        op = in_port.node

        # Disconnect FQ from input and reconnect outputs to input node
        self.remove_node_and_reset_connections(graph, fq, in_port)

        # Insert FQ operations for split input
        return insert_fake_quantize(graph, get_node_inputs(op)[0], [0])

    def remove_duplication(self, graph: Graph, fq: Node) -> []:
        # Keep only input operation
        fq.out_port(0).get_connection().set_source(fq.in_port(0).get_source())
        fq.in_port(0).disconnect()
        graph.remove_node(fq.id)
        return []

    def check_split_concat(self, node):
        if node.type != 'Concat':
            return False
        upper_ops = get_node_inputs(node)
        return upper_ops[0].type not in SPLIT_OPERATIONS \
               and len({upper_op.name for upper_op in upper_ops}) == 1 \
               and len({up_down.name for up_down in get_all_node_outputs(upper_ops[0])}) == 1

    jump_single_branch_ops = ['ReduceMax', 'MaxPool', 'Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Interpolate',
                              'Split', 'Crop', 'ReduceMean', 'AvgPool', 'Result', 'Tile', 'Transpose', 'StridedSlice',
                              'VariadicSplit', 'ShuffleChannels', 'Broadcast', 'Minimum', 'Maximum', 'DepthToSpace']
    remove_duplication_ops = ['FakeQuantize', 'Parameter']
    jump_multi_branch_ops = 'Concat'
    jump_multi_branch_ops_except_const = ['Pad', 'ConvertLike']
    jump_split_concat_ops = ('Split', 'VariadicSplit', 'Concat')
    map_op_to_fn = {
        **dict(zip(jump_single_branch_ops, cycle([jump_to_first_input]))),
        **dict(zip(
            remove_duplication_ops, cycle([remove_duplication]))),
        jump_multi_branch_ops: jump_to_all_inputs,
        **dict(zip(
            jump_multi_branch_ops_except_const, cycle([jump_to_all_branch_except_const]))),
        jump_split_concat_ops: jump_over_split_concat
    }

    def delete_fq_non_quantizable_node_precision(self, graph):
        fq_removal = RemoveFakeQuantize()
        fq_removal.quantize_agnostic_operations = self.quantize_agnostic_operations
        fq_removal.quantize_operations = self.quantize_operations
        node_int_fq = []
        fq_queue = deque(sorted(graph.get_op_nodes(type='FakeQuantize'), key=lambda x: x.name))
        while fq_queue:
            fq = fq_queue.popleft()
            if fq.in_port(0).get_source() is not None and fq.in_port(0).get_source().is_data_type_defined():
                type_node = fq.in_port(0).get_source().get_data_type()
                if type_node in (np.int32, np.int64, bool):
                    node_int_fq.append(fq.name)
                    fq_removal.find_and_remove_node(graph, fq.name)

    @property
    def quantize_inputs(self):
        return getattr(self, '_quantize_inputs', False)

    @quantize_inputs.setter
    def quantize_inputs(self, value):
        setattr(self, '_quantize_inputs', value)

    @property
    def quantize_operations(self):
        return getattr(self, '_quantize_operations', None)

    @quantize_operations.setter
    def quantize_operations(self, value):
        setattr(self, '_quantize_operations', value)

    @property
    def quantize_agnostic_operations(self):
        return getattr(self, '_quantize_agnostic_operations', [])

    @quantize_agnostic_operations.setter
    def quantize_agnostic_operations(self, value):
        setattr(self, '_quantize_agnostic_operations', value)
        for op in value:
            if op['type'] not in self.map_op_to_fn:
                raise RuntimeError('FakeQuantizePropagation could not support operation {}'.format(op))

    def find_and_replace_pattern(self, graph: Graph):
        fq_queue = deque(sorted(graph.get_op_nodes(type='FakeQuantize'), key=lambda x: x.name))
        skip_ascent_map = self._create_skip_ascent_map(graph)
        # Iterate over FakeQuantize operations and push them on top while it's possible
        while fq_queue:
            # Get FakeQuantize from queue and it's input node
            fq = fq_queue.popleft()

            # In case if we already touched this FakeQuantize it could be disconnected from the main graph
            if fq.in_port(0).disconnected():
                continue

            input_node = fq.in_port(0).get_source().node
            input_type = input_node.type
            output_node = nu.get_node_output(fq, 0)[0]
            output_type = output_node.type

            # Check that input type is allowed from jumping over
            m_op = find_operation_matches(self.quantize_agnostic_operations, input_node)
            is_scaleshift = output_type == 'Multiply' and nu.get_node_output(output_node, 0)[0].type == 'Add'
            if len(m_op) > 1:
                raise RuntimeError(
                    'FakeQuantizePropagation matched several callback functions for operation {}'.format(input_node))
            if input_type not in self.remove_duplication_ops and \
                    skip_ascent_map[input_node.name]:
                continue
            if m_op \
                    or input_type == 'FakeQuantize' \
                    or (input_type == 'Parameter'
                            and is_scaleshift
                            and not self.quantize_inputs):
                input_name = input_node.name
                if self.check_split_concat(input_node):
                    input_parent_name = get_node_inputs(input_node)[0].name
                    if not skip_ascent_map[input_parent_name]:
                        input_type = ('Split', 'VariadicSplit', 'Concat')
                        input_name = (input_node.name, input_parent_name)
                if input_type == 'FakeQuantize':
                    if fq['fq_config_priority'] > input_node['fq_config_priority']:
                        input_node['fq_group'] = fq['fq_group']
                        input_node['fq_config_priority'] = fq['fq_config_priority']
                    for fq_config in fq['fq_configs']:
                        if fq_config not in input_node['fq_configs']:
                            input_node['fq_configs'].append(fq_config)
                    logger.debug('FQ %s extended with %s configs', input_name, fq.name)
                logger.debug('FQ %s jumped over %s (%s)', fq.name, input_type, input_name)

                callback = self.map_op_to_fn[input_type]
                new_fq = callback(self, graph, fq)

                # Update queue with new (moved) FQ operations
                if isinstance(new_fq, list):
                    for fq in new_fq:
                        fq_queue.appendleft(fq)
                elif isinstance(new_fq, Node) and new_fq.type == 'FakeQuantize':
                    fq_queue.appendleft(new_fq)
                else:
                    raise RuntimeError(
                        'Unsupported response ({}) from callback {}.'.format(
                            new_fq, self.map_op_to_fn[input_type]))

    def _create_skip_ascent_map(self, graph: Graph) -> {}:

        def _is_node_skippable(node, skip_ascent_map):
            skippable_ops = [*self.jump_single_branch_ops, self.jump_multi_branch_ops, 'FakeQuantize',
                             *self.jump_multi_branch_ops_except_const]

            def sink_fn(op):
                out = []
                if op.type != 'FakeQuantize':
                    out = [n for n in get_all_node_outputs(op) if n.type != 'ShapeOf']
                return out

            def source_fn(op):
                return [p for p in get_node_inputs(op)
                        if p and p.type not in ['FakeQuantize', 'Const'] and p.name != node.name]

            def is_multibranch_fn(op):
                return op.type == self.jump_multi_branch_ops

            def has_fake_quantize_fn(op):
                return op.type == 'FakeQuantize'

            def not_skippable_op_fn(op):
                return op.type not in skippable_ops

            def process_multibranch_descendants(criteria):
                _skip_multibranch_ascent_ops = {}
                for name in criteria[is_multibranch_fn]:
                    if name in skip_ascent_map:
                        _skip_multibranch_ascent_ops[name] = skip_ascent_map[name]
                    else:
                        _skip_multibranch_ascent_ops[name] = _is_node_skippable(
                            ge.get_node_by_name(graph, name, recursively=False), skip_ascent_map)
                skip_ascent_map.update(_skip_multibranch_ascent_ops)
                return any(_skip_multibranch_ascent_ops.values())

            def process_multibranch(op):
                if not traverse_graph(op, source_fn, not_skippable_op_fn)[0]:
                    return False
                res, criteria = traverse_graph(
                    op, sink_fn, not_skippable_op_fn,
                    [is_multibranch_fn, has_fake_quantize_fn])
                if res or has_fake_quantize_fn not in criteria:
                    return True
                if is_multibranch_fn in criteria:
                    return process_multibranch_descendants(criteria)
                return has_fake_quantize_fn not in criteria

            def process_singlebranch(op):
                res, criteria = traverse_graph(
                    op, sink_fn, not_skippable_op_fn, is_multibranch_fn)
                if criteria:
                    return process_multibranch_descendants(criteria)
                return res

            def stop_fn(op):
                if op.type not in skippable_ops:
                    return True
                if op.name == node.name:
                    return False
                if is_multibranch_fn(op):
                    return process_multibranch(op)
                return process_singlebranch(op)

            return traverse_graph(node, sink_fn, stop_fn)[0]

        skip_ascent = {}
        for op in graph.get_op_nodes():
            if 'skipped' in op and op['skipped']:
                skip_ascent[op.name] = True
            if op.name not in skip_ascent:
                skip_ascent[op.name] = _is_node_skippable(op, skip_ascent)

        return skip_ascent


class FakeQuantizeOptimization(BackReplacementPattern):

    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        for op in sorted(graph.get_op_nodes(), key=lambda x: x.name):
            for _, out_port in op.out_ports().items():
                if out_port.disconnected():
                    continue
                # Get all consumers that are FakeQuantize
                fq_consumers = [in_port.node for in_port in out_port.get_destinations()
                                if in_port.node.type == 'FakeQuantize' and in_port.idx == 0]
                fq_consumers = sorted(fq_consumers, key=lambda x: x.name)
                # Keep only first FakeQuantize and disconnect other
                for fq in fq_consumers[1:]:
                    for fq_config in fq['fq_configs']:
                        if fq_config not in fq_consumers[0]['fq_configs']:
                            fq_consumers[0]['fq_configs'].append(fq_config)
                    logger.debug('Removed useless FakeQuantize {}'.format(fq.name))
                    fq.in_port(0).disconnect()
                    fq.out_port(0).get_connection().set_source(fq_consumers[0].out_port(0))


class RemoveFakeQuantize:
    def find_and_remove_node(self, graph, node_name, force=False):
        node = ge.get_node_by_name(graph, node_name, recursively=False)
        if not node:
            return [], []

        if force:
            self.disconnect_fq_node(node)
            return [node_name], []

        nodes_to_cut, ops_in_orig_prec = self.find_fq_nodes_to_cut(node)
        for fq_node in nodes_to_cut:
            self.disconnect_fq_node(fq_node)
            self.undo_renaming(graph, fq_node)

        for op in ops_in_orig_prec:
            if op.type in ('Convolution', 'MatMul'):
                self.undo_bias_correction(op)
                self.undo_weights_rescaling(op)

        return [fq_node.name for fq_node in nodes_to_cut], [op.name for op in ops_in_orig_prec]

    def find_fq_nodes_to_cut(self, node):
        def parse_node_relatives(node, is_parents):
            if not node:
                return

            if find_operation_matches(self.quantize_operations, node):
                ops_to_return_in_orig_prec.add(node)

            seen_list = seen_parents if is_parents else seen_children
            relatives_ports = nu.get_node_input_ports(node) if is_parents else nu.get_node_output_ports(node)
            relatives_ports = [p for p in relatives_ports if p]
            for relative_port in relatives_ports:
                relative = relative_port.node
                if relative.type == 'FakeQuantize':
                    if is_parents:
                        if relative.name in seen_children:
                            continue
                        if relative not in to_cut:
                            to_cut.append(relative)
                        to_see_children.append(relative)
                    else:
                        seen_children.append(relative.name)
                elif relative.type != 'Const' and relative_port.data.get_value() is None:
                # Here, propagation to KSO subgraphs is blocked by checking the data value
                # which is None for input data propagated nodes.
                    if relative.name not in seen_parents:
                        to_see_parents.append(relative)
                    if relative.name not in seen_children and \
                            find_operation_matches(self.quantize_agnostic_operations, relative):
                        to_see_children.append(relative)
            seen_list.append(node.name)

        seen_children, seen_parents = [], []
        to_see_children, to_see_parents = [node], []
        to_cut = [node]
        ops_to_return_in_orig_prec = set()

        while to_see_parents or to_see_children:
            if to_see_children:
                node = to_see_children.pop()
                parse_node_relatives(node, is_parents=False)
            if to_see_parents:
                node = to_see_parents.pop()
                parse_node_relatives(node, is_parents=True)

        return to_cut, ops_to_return_in_orig_prec

    def disconnect_fq_node(self, fq_node):
        parent_node_port = fq_node.in_port(0).get_source()
        parent_node = parent_node_port.node
        fq_node.in_port(0).disconnect()
        for port in fq_node.out_ports().values():
            port.get_connection().set_source(parent_node_port)
        if parent_node.type == 'Const':
            parent_node['need_shape_inference'] = True

    def optimize_for_gp_hw(self, graph, target_device):
        """
        Removing redundant FQs before operation Add for SPR(CPU) platform
        """
        def _walk_for_branch(node):
            input_node = node
            delete_const = lambda node: ([op for op in node if op is not None and op.type != 'Const'])
            while True:
                input_node = get_node_inputs(input_node)
                input_node = delete_const(input_node)
                if len(input_node) > 1:
                    return False
                input_node = input_node[0]
                if input_node.type in ['Convolution', 'GroupConvolution', 'MatMul']:
                    return True

        def _check_const_input(node):
            input_node = get_node_inputs(node)[0]
            return nu.check_const_input(input_node)

        def delete_one_fq(inputs_node):
            fq_1, fq_2 = inputs_node
            if len(get_all_node_outputs(fq_1)) > 1 \
                and len(get_all_node_outputs(fq_2)) == 1 and _check_const_input(fq_2):
                self.disconnect_fq_node(fq_2)
                return
            if _walk_for_branch(fq_1) and _walk_for_branch(fq_2):
                if np.prod(nu.get_output_shape(fq_1, 0)) >= np.prod(nu.get_output_shape(fq_2, 0)):
                    self.disconnect_fq_node(fq_1)
                else:
                    self.disconnect_fq_node(fq_2)
                return

        special_target_device = ['CPU_SPR']
        if target_device not in special_target_device:
            return

        check_is_inputs_fq = lambda node: all([op.type == 'FakeQuantize' for op in node])
        for op in get_nodes_by_type(graph, ['Add']):
            if not nu.check_const_input(op):
                inputs_node = np.array(get_node_inputs(op))
                count_outputs_node = np.array([len(get_all_node_outputs(node)) for node in inputs_node])
                indices = count_outputs_node.argsort()[::-1]
                inputs_node = inputs_node[indices]
                if check_is_inputs_fq(inputs_node):
                    delete_one_fq(inputs_node)

    @staticmethod
    def undo_bias_correction(conv_node):
        bias_node = nu.get_bias_for_node(conv_node)
        if bias_node and 'original_bias' in conv_node:
            nu.set_bias_for_node(conv_node, conv_node['original_bias'])

    @staticmethod
    def undo_weights_rescaling(conv_node):
        weights_node = nu.get_node_input(conv_node, 1)
        if weights_node.type == 'FakeQuantize':
            weights_node = nu.get_node_input(weights_node, 0)
        if 'scaling_factor' in conv_node:
            nu.set_node_value(weights_node, nu.get_node_value(weights_node) * conv_node['scaling_factor'])
        if 'wbc_mean_shift' in conv_node:
            original_weights = (nu.get_node_value(weights_node) - conv_node['wbc_mean_shift']) / \
                conv_node['wbc_variance_shift']
            nu.set_node_value(weights_node, original_weights)

    @staticmethod
    def undo_renaming(graph, fq_node):
        if 'orig_fq_name' in fq_node:
            node = ge.get_node_by_name(graph,
                                       '{fq_name}/pre_fq_input'.format(fq_name=fq_node.fullname),
                                       recursively=False)
            rename_node(node, node['orig_node_name'])
            rename_node(fq_node, fq_node['orig_fq_name'])

    @property
    def quantize_agnostic_operations(self):
        return getattr(self, '_quantize_agnostic_operations', [])

    @quantize_agnostic_operations.setter
    def quantize_agnostic_operations(self, value):
        setattr(self, '_quantize_agnostic_operations', value)

    @property
    def quantize_operations(self):
        return getattr(self, '_quantize_operations', [])

    @quantize_operations.setter
    def quantize_operations(self, value):
        setattr(self, '_quantize_operations', value)


class SpecialBlocksMarker:
    @staticmethod
    def mark_block_nodes(check_pattern_fn, _, match):
        if check_pattern_fn and check_pattern_fn(match):
            return
        match_list = [match[node] for node in match if node not in ['input', 'output']]
        for node in match_list:
            if node.kind == 'op':
                node['skipped'] = True

    def mark_ignored_blocks(self, graph, target_device):
        def mark_ignored_blocks_(patterns):
            for types_list in patterns:
                for pattern in patterns[types_list]:
                    if isinstance(pattern, tuple):
                        pattern, check_pattern_fn = pattern
                        mark_fn = partial(self.mark_block_nodes, check_pattern_fn)
                    else:
                        mark_fn = partial(self.mark_block_nodes, None)
                    apply_pattern(
                        graph,
                        nodes=pattern['nodes'],
                        edges=pattern['edges'],
                        action=mark_fn
                    )

        def mark_detection_output_blocks_(graph):
            det_out_finals = ge.get_nodes_by_type(graph, [op['type'] for op in DETECTION_OUTPUT_FINAL_TYPES])
            stop_propagation_types = [op['type'] for op in OPERATIONS_WITH_WEIGHTS]
            stop_propagation_types.append('Const')

            def move_fn(op):
                return [node for node in nu.get_node_inputs(op) if
                        node is not None and node.type not in stop_propagation_types]

            def stop_fn(op):
                op['skipped'] = True
                return False

            for det_out_final in det_out_finals:
                traverse_graph(det_out_final, move_fn, stop_fn)

        mark_ignored_blocks_(get_hw_aware_ignored_patterns(target_device))
        mark_ignored_blocks_(get_ignored_patterns())
        mark_detection_output_blocks_(graph)


class MatMulPreprocessing(BackReplacementPattern):
    enabled = False

    def pattern(self):
        return dict(
            nodes=[
                ('matmul', {'kind': 'op', 'type': 'MatMul'})
            ],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        matmul = match['matmul']
        input_const = nu.get_node_input(matmul, 1)
        if input_const.type == 'Const' \
                and len(input_const.shape) > 1 \
                and not matmul['transpose_b']:
            matmul['transpose_b'] = not matmul['transpose_b']
            matmul_w = nu.get_node_input(matmul, 1)
            matmul_w_value = nu.get_node_value(matmul_w)
            matmul_w_value = np.moveaxis(matmul_w_value, -2, -1)
            nu.set_node_value(matmul_w, matmul_w_value)
        return graph


class IgnoreShapeSubgraph(BackReplacementPattern):
    enabled = False

    def pattern(self):
        return dict(
            nodes=[
                ('shape', {'kind': 'op', 'type': 'ShapeOf'})
            ],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        shape = match['shape']
        all_shape_nodes = find_shape_subgraph_endpoints([shape.out_port(0)])
        for node in all_shape_nodes:
            node['skipped'] = True
        return graph


class ModelPreprocessor(BackReplacementPattern):
    """
    Performing equivalent graph transformation needed for further work.
    """
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        MatMulPreprocessing().find_and_replace_pattern(graph)
        IgnoreShapeSubgraph().find_and_replace_pattern(graph)
        InsertBiasNode().insert_null_biases(graph)


class InsertBiasNode:

    def insert_null_biases(self, graph: Graph):
        # Get nodes by type only for Convolutions instead of OPERATIONS_WITH_BIAS
        for node in ge.get_nodes_by_type(graph, ['Convolution']):
            if not nu.get_bias_for_node(node):
                create_bias_node(graph, node)


class FakeQuantizeNameSwapper(BackReplacementPattern):
    """
    Performing equivalent graph transformation needed for further work.
    """
    enabled = False

    def rename_fqs_in_the_end(self, graph: Graph):
        def change_names(_, match):
            fq_node = match['fq']
            input_node = get_node_input(fq_node, 0)
            new_fq_name = copy(input_node.name)
            if 'orig_node_name' in input_node:
                new_fq_name = copy(input_node['orig_node_name'])

            input_node_outputs = get_all_node_outputs(input_node)
            if len(input_node_outputs) > 1 and all([op.type == 'FakeQuantize' for op in input_node_outputs]):
                new_fq_name += '.{}'.format(fq_node.in_port(0).get_source().idx)

            fq_node['orig_fq_name'] = copy(fq_node.name)

            if 'orig_node_name' not in input_node:
                input_node['orig_node_name'] = copy(input_node.name)
                rename_node(input_node, f'{input_node.name}/pre_fq_input')
            rename_node(fq_node, new_fq_name)

        pattern = get_fq_result_pattern()
        apply_pattern(
            graph,
            nodes=pattern['nodes'],
            edges=pattern['edges'],
            action=change_names
        )


def create_bias_node(graph: Graph, src_node):
    logger.debug('Creating new bias for {}'.format(src_node.name))
    destination_ports = []
    for dest_port in src_node.out_port(0).get_destinations():
        destination_ports.append(dest_port)

    # Create Add and constant with zero bias
    bias_shape = src_node.out_port(0).data.get_shape()
    add_bias_shape = [1] * len(bias_shape)
    add_bias_shape[1] = bias_shape[1]
    weights = get_weights_for_node(src_node)
    bias_dtype = np.float32
    if weights and weights.out_port(0).is_data_type_defined():
        bias_dtype = weights.out_port(0).get_data_type()
    add_bias = Const(graph,
                     {'value': np.zeros(add_bias_shape, dtype=bias_dtype),
                      'shape': add_bias_shape,
                      'need_shape_inference': True
                      }).create_node()
    add_op = Add(graph, {'name': src_node.name + '/add_',
                         'need_shape_inference': True}).create_node()

    # Connect Const to Add node
    add_op.in_port(1).connect(add_bias.out_port(0))

    # Reconnect src_node -> output to src_node -> Add -> output
    src_node.out_port(0).disconnect()
    src_node.out_port(0).get_connection().set_destination(add_op.in_port(0))

    for destination_port in destination_ports:
        add_op.out_port(0).connect(destination_port)
    if bias_dtype != np.float32:
        add_bias.out_node(0)['Insert_Convert_operation_after'] = True


def create_fake_quantize_node(graph: Graph, name, data_type=np.float32, **kwargs):
    fq = FakeQuantize(graph, {'name': name, 'levels': 0,
                              'stop_value_propagation': True, **kwargs}).create_node()

    input_low = Const(graph, {'value': np.array(0.0, dtype=data_type)}).create_node()
    input_height = Const(graph, {'value': np.array(0.0, dtype=data_type)}).create_node()
    output_low = Const(graph, {'value': np.array(0.0, dtype=data_type)}).create_node()
    output_height = Const(graph, {'value': np.array(0.0, dtype=data_type)}).create_node()

    input_low.out_port(0).connect(fq.in_port(1))
    input_height.out_port(0).connect(fq.in_port(2))
    output_low.out_port(0).connect(fq.in_port(3))
    output_height.out_port(0).connect(fq.in_port(4))

    input_low.infer(input_low)
    input_height.infer(input_height)
    output_low.infer(output_low)
    output_height.infer(output_height)

    return fq


def insert_fake_quantize(graph, node, ports=None, names=None, fq_types=None, hw_config=None, input_priority_types=[]):
    blobs_as_inputs_nodes_type = ['Convolution', 'Deconvolution', 'MatMul']

    port_name = None
    if ports is not None and names is not None:
        port_name = dict(zip(ports, names))

    fq_type = None
    if fq_types is not None and ports is not None:
        fq_type = dict(zip(ports, fq_types))

    new_fq = []
    for idx, port in node.in_ports().items():
        if port.disconnected():
            continue

        # Temporary WA while blobs_as_inputs option isn't work properly
        if node.type in blobs_as_inputs_nodes_type:
            if 'bin' in node.in_edges()[idx]:
                del node.in_edges()[idx]['bin']

        if ports is not None and idx not in ports:
            continue

        # This condition blocks FQ insertion after the keep_shape_ops (KSO) generated sub-graph
        # to avoid quantization of integer-like tensors
        in_port_type = port.get_source().node.type
        if in_port_type != 'Const' and port.data.get_value() is not None:
            continue

        is_weights = in_port_type == 'Const'

        name = 'fq_weights' if is_weights else 'fq_input'
        if port_name is not None and idx in port_name:
            name = port_name[idx]

        port_data_type = nu.get_node_data_type(node, idx)
        port_data_type = port_data_type if port_data_type else np.float32
        # Create FakeQuantize operations
        fq_group = 'weights' if is_weights else 'activations'
        if fq_type is not None and idx in fq_type:
            fq_group = fq_type[idx]

        fq_configs = []
        if hw_config is not None:
            node_type = get_hardware_config_operation_type(node, list(hw_config.keys()))
            if hw_config[node_type]:
                fq_configs = hw_config[node_type][fq_group]

        if node_type in input_priority_types:
            fq_config_priority = 2
        else:
            fq_config_priority = 1

        fq_options = {
            'fq_group': fq_group,
            'fq_configs': copy(fq_configs),
            'fq_config_priority': fq_config_priority
        }

        fq_name = '{node_name}/{name}_{idx}'.format(node_name=node.name, name=name, idx=idx)
        fq_input = create_fake_quantize_node(graph, fq_name, port_data_type, **fq_options)
        # Insert FakeQuantize after input
        if node.type == 'Result':
            in_port = port.get_source()
            port.get_connection().set_source(fq_input.out_port(0))
            in_port.connect(fq_input.in_port(0))
        else:
            port.get_connection().set_destination(fq_input.in_port(0))
            fq_input.out_port(0).connect(port)

        fq_input.infer(fq_input)

        new_fq.append(fq_input)
    return new_fq


def insert_output_fake_quantize(graph, node, hw_config=None, ignored_params=None):
    activation_nodes_type = ['Power', 'Sigmoid', 'Tanh', 'ReLU', 'PReLU',
                            'Clamp', 'Log', 'Abs', 'Exp', 'Sign', 'SoftSign']

    new_fq = []
    for out_port_id, port in node.out_ports().items():
        if port.disconnected():
            continue

        next_ports = port.get_destinations()
        for next_port_id, next_port in enumerate(next_ports):
            next_node = next_port.node

            if next_node.type == 'ShapeOf':
                continue

            if ignored_params is not None and next_node.type != 'FakeQuantize' \
                    and is_ignored(ignored_params, next_node):
                continue

            fq_name = '{node_name}/fq_output_{out_port_id}_{next_port_id}'.format(node_name=node.name,
                                                                                  out_port_id=out_port_id,
                                                                                  next_port_id=next_port_id)
            fq_configs = hw_config[node.type]['outputs'] if hw_config is not None and hw_config[node.type] else []

            fq_config_priority = 0
            if node.type in activation_nodes_type + ['Parameter']:
                fq_config_priority = 1
            else:
                fq_config_priority = 0

            fq_options = {
                'fq_group': 'outputs',
                'fq_configs': copy(fq_configs),
                'fq_config_priority': fq_config_priority
            }
            fq_output = create_fake_quantize_node(graph, fq_name, **fq_options)

            in_port = next_port.get_source()
            next_port.get_connection().set_source(fq_output.out_port(0))
            in_port.connect(fq_output.in_port(0))
            fq_output.infer(fq_output)
            new_fq.append(fq_output)

    return new_fq


def traverse_graph(node, move_fn, stop_criteria_fn=None, criteria_fns=None):
    """ Traverse through graph dependent on move_fn
    :param node: node to start floating or sinking with some rule
    :param move_fn: function to get relatives (children, parents or some subset of them)
    to make traverse through graph. Function should have node as argument.
    You can make traverse up/down or whatever you want using this function.
    :param stop_criteria_fn: function to stop traversing and return boolean result.
    Function should have node as argument.
    :param criteria_fns: list of functions or just function with specified criteria for nodes.
    Returns True if criteria was satisfied at least at one node, False otherwise
    :return pair of values. The first one is a boolean value. In case stop criteria was satisfied
    the value is True, False otherwise.
    The second value is a dict of criteria values. The keys are function itself, node names where
    criteria was satisfied is used as dict values.
     """

    if criteria_fns and callable(criteria_fns):
        criteria_fns = [criteria_fns]

    criteria_res = {}

    def apply_criteria_fn(n):
        if not criteria_fns:
            return
        for fn in criteria_fns:
            node_name = n.name if fn(n) else None
            if node_name:
                if fn in criteria_res:
                    criteria_res[fn].append(node_name)
                else:
                    criteria_res[fn] = [node_name]

    def stop_traverse_fn(n):
        if stop_criteria_fn:
            return stop_criteria_fn(n)
        return False

    queue, visited = [node], []

    while queue:
        current_node = queue.pop(0)
        if current_node.name in visited:
            continue

        relatives = move_fn(current_node)
        visited.append(current_node.name)

        if relatives:
            for r in relatives:
                apply_criteria_fn(r)
                if stop_traverse_fn(r):
                    return True, criteria_res
        queue += relatives
    return False, criteria_res


def compress_weights(model: Graph):
    """Apply transformations to save model weights to INT8."""
    add_removed_converts(model)
    CompressQuantizeWeights().find_and_replace_pattern(model)
    model.clean_up()
    ZeroPointOptimizer().find_and_replace_pattern(model)
    model.clean_up()
    ForceStrictPrecision().find_and_replace_pattern(model)
    model.clean_up()


def get_next_in_ports(in_port: Port) -> Set[Port]:
    next_in_ports = set()
    for out_port in in_port.node.out_ports().values():
        next_in_ports.update(out_port.get_destinations())
    return next_in_ports


def find_shape_subgraph_endpoints(out_ports: List[Port], visited: set = None) -> Set[Port]:
    """
    Searches for input ports of data dependent operations starting from output ports passed to the function.
    Condition for data dependent operations is absence of node output value.

    :param out_ports: list of output ports to start search from
    :param visited: set of input ports that were visited to avoid visiting them more than once
    :return: set of all nodes that are part of shape calculating subgraph
    """
    if visited is None:
        visited = set()

    deque_of_in_ports = deque()
    for out_port in out_ports:
        deque_of_in_ports.extend(out_port.get_destinations())

    end_points_in_ports = set()
    visited_nodes = set()
    while deque_of_in_ports:
        in_port = deque_of_in_ports.popleft()
        if in_port in visited:
            continue

        next_in_ports = get_next_in_ports(in_port)
        if any([port.data.get_value() is None for port in next_in_ports]):
            end_points_in_ports.add(in_port)
        else:
            deque_of_in_ports.extend(next_in_ports)
            visited_nodes.add(in_port.node)
        visited.add(in_port)
    return visited_nodes


def remove_converts(graph: Graph):
    for op in graph.get_op_nodes():
        if op.type == 'Convert':
            source_op = op.in_port(0).get_source().node
            if source_op.type == 'Const' and source_op.data_type == np.float16:
                # Get access to data node after Convert operation and set Insert_Convert_operation_after
                # to restore Convert operation later
                op.out_node(0)['Insert_Convert_operation_after'] = True
                # Mark Const and Convert operation to fold them
                source_op['need_shape_inference'] = True
                op.out_node(0)['old_rt_info'] = op['rt_info']
                op['stop_value_propagation'] = False
        op['need_shape_inference'] = True
    graph.clean_up()


def add_removed_converts(graph: Graph):
    for data_node_name in graph.get_nodes_with_attributes(Insert_Convert_operation_after=True):
        data_node = Node(graph, data_node_name)
        # Get access to Const node connected to data node
        const_op = data_node.in_node(0)

        if const_op.type != 'Const':
            logger.debug('Error when try to insert Convert operation after {} with {} type'.\
                format(const_op.soft_get('name'), const_op.soft_get('type')))
            continue

        if const_op.data_type != np.float32:
            logger.debug('Error when try to insert Convert operation after Const: {}'.\
                format(const_op.soft_get('name')))
            continue

        convert_op = Cast(graph, {'dst_type': np.float32,
                                  'name': const_op.name + '/restored_convert',
                                  'stop_value_propagation': True}).create_node()

        # Insert Convert operation after Const operation
        const_op.out_port(0).get_connection().insert_node(convert_op)
        convert_op.out_node().value = None
        convert_op['rt_info'] = data_node['old_rt_info']

        # Convert Const value to FP16 to make types in graph consistent
        const_op.value, _, _ = convert_blob(const_op.value, np.float16)
        const_op.infer(const_op)
