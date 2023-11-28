# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pattern_builder import PatternBuilder
from .node_utils import get_node_input, get_all_node_outputs
from ..graph.special_operations import OPERATIONS_WITH_BIAS, ELTWISE_TYPES, is_eltwise


def check_fused_scale_shift_patterns(match):
    node = [match[node] for node in match if match[node].kind == 'op' and match[node].type == 'Multiply'][0]
    return check_for_branching(node)


def get_fused_scale_shift_patterns():
    return [pattern.insert_scaleshift().pattern for pattern in fc_conv_fused_basis_patterns()]


def check_fused_op_const_patterns(match):
    nodes = [match[node] for node in match if match[node].kind == 'op' and is_eltwise(match[node])]
    return any([check_for_branching(node) for node in nodes])


def get_fused_op_const_pattern():
    return [pattern.append_op_const(lambda x: x in ELTWISE_TYPES, 'eltwise').pattern
            for pattern in fc_conv_fused_basis_patterns()]


def fc_conv_fused_basis_patterns():
    base_pattern = PatternBuilder().insert_conv_fc(name='input')

    pattern_with_bias = PatternBuilder()
    pattern_with_bias.insert_conv_fc(name='input')
    pattern_with_bias.insert_bias()

    pattern_with_activation = PatternBuilder()
    pattern_with_activation.insert_conv_fc(name='input')
    pattern_with_activation.insert_activation()

    pattern_with_both = PatternBuilder()
    pattern_with_both.insert_conv_fc(name='input')
    pattern_with_both.insert_bias()
    pattern_with_both.insert_activation()

    return [
        base_pattern,
        pattern_with_bias,
        pattern_with_activation,
        pattern_with_both
    ]


def check_for_branching(node):
    parent_node_type = None
    # This list of types ends propagation through constant and FC nodes
    skip_types = [op['type'] for op in OPERATIONS_WITH_BIAS]
    skip_types.append('Const')
    while parent_node_type not in skip_types:
        parent_node = get_node_input(node, 0)
        parent_node_type = parent_node.type
        if len(get_all_node_outputs(parent_node)) > 1:
            return True
        node = parent_node
    return False


def get_clamp_mult_const_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op(op_type='Clamp', op_name='clamp')
    pattern.insert_multiply_const()
    return pattern.set_name('hswish_activation_clamp_multiply').pattern


def get_assign_result_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Assign', 'input')
    pattern.append_single_op('Result', 'result')
    return pattern.set_name('assign_result').pattern


def get_fq_result_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('FakeQuantize', 'fq')
    pattern.append_single_op('Result', 'result')
    return pattern.set_name('fq_result').pattern


# Self-attention block in vision transformers (Swin, Twins, ViTPose)
def get_softmax_reshape_transpose_gather_matmul_pattern():
    pattern = PatternBuilder()
    pattern_2 = PatternBuilder()
    softmax_out = pattern.append_single_op('SoftMax', 'softmax').get_last_node()
    pattern_2.append_single_op('Add', 'add').get_last_node()
    pattern_2.append_op_const('Reshape', 'reshape')
    pattern_2.append_single_op('Transpose', 'transpose').get_last_node()
    gather_out = pattern_2.append_single_op('Gather', 'gather').get_last_node()
    pattern.pattern['nodes'] += pattern_2.pattern['nodes']
    pattern.pattern['edges'] += pattern_2.pattern['edges']
    pattern.insert_single_op([softmax_out, gather_out], None, 'MatMul', 'matmul')
    return pattern.set_name('softmax_reshape_transpose_gather_matmul').pattern


def get_softmax_reshape_transpose_matmul_pattern():
    pattern = PatternBuilder()
    pattern_2 = PatternBuilder()
    softmax_out = pattern.append_single_op('SoftMax', 'softmax').get_last_node()
    pattern_2.append_single_op('Add', 'add').get_last_node()
    pattern_2.append_op_const('Reshape', 'reshape')
    transp_out = pattern_2.append_single_op('Transpose', 'transpose').get_last_node()
    pattern.pattern['nodes'] += pattern_2.pattern['nodes']
    pattern.pattern['edges'] += pattern_2.pattern['edges']
    pattern.insert_single_op([transp_out, softmax_out], None, 'MatMul', 'matmul')
    return pattern.set_name('softmax_reshape_transpose_matmul').pattern
