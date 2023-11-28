# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pattern_builder import PatternBuilder
from .special_operations import OPERATIONS_WITH_WEIGHTS

IGNORED_PATTERNS = {}


def get_ignored_patterns():
    return IGNORED_PATTERNS


def registry_ignore_patterns(key):
    def add_ignored_patterns(func):
        if key not in IGNORED_PATTERNS:
            IGNORED_PATTERNS[key] = []
        IGNORED_PATTERNS[key].append(func())
    return add_ignored_patterns


@registry_ignore_patterns('activations')
def create_swish_pattern():
    pattern = PatternBuilder()
    pattern.insert_conv_fc(name='input')
    pattern.insert_bias()
    pattern.insert_swish()
    return pattern.set_name('swish_activation').pattern


@registry_ignore_patterns('blocks')
def create_se_pattern():
    """
    Removing this pattern can drop accuracy after quantization of model w/ SE-blocks
    """
    pattern = PatternBuilder()
    pattern.insert_se(start_name='input', end_name='output')
    return pattern.set_name('se_block').pattern


@registry_ignore_patterns('blocks')
def create_se_swish_pattern():
    """
    Removing this pattern can drop accuracy after quantization of model w/ SE-blocks
    """
    pattern = PatternBuilder()
    pattern.insert_se(start_name='input', end_name='output', is_swish=True)
    return pattern.set_name('se_block_swish_activation').pattern


@registry_ignore_patterns('blocks')
def create_biased_op_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op(lambda x: x in [op['type'] for op in OPERATIONS_WITH_WEIGHTS], 'input')
    pattern.insert_bias()
    return pattern.set_name('operation_with_bias').pattern


@registry_ignore_patterns('blocks')
def create_scaleshift_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Multiply', 'input')
    pattern.insert_add_const()
    return pattern.set_name('scale_shift_add').pattern


@registry_ignore_patterns('blocks')
def create_add_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Add', 'input')
    pattern.insert_multiply_const()
    pattern.insert_add_const()
    pattern.append_single_op('Result', 'result')
    return pattern.set_name('add_scale_shift').pattern


@registry_ignore_patterns('blocks')
def create_mvn_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('MVN', 'input')
    pattern.insert_scaleshift()
    return pattern.set_name('mvn_scale_shift').pattern


@registry_ignore_patterns('blocks')
def create_normalize_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('NormalizeL2', 'input')
    pattern.append_single_op('Multiply', 'multiply')
    return pattern.set_name('normalize_l2').pattern


@registry_ignore_patterns('inputs')
def create_input_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.insert_scaleshift()
    return pattern.set_name('input_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_transpose_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_scaleshift()
    return pattern.set_name('input_transpose_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_convert_transpose_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Convert', 'convert')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_scaleshift()
    return pattern.set_name('input_convert_transpose_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.insert_add_const()
    return pattern.set_name('input_add').pattern


@registry_ignore_patterns('inputs')
def create_input_subtract_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Subtract', 'subtract')
    return pattern.set_name('input_subtract').pattern


@registry_ignore_patterns('inputs')
def create_input_transpose_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_add_const()
    return pattern.set_name('input_transpose_add').pattern


@registry_ignore_patterns('inputs')
def create_input_convert_transpose_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Convert', 'convert')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_add_const()
    return pattern.set_name('input_convert_transpose_add').pattern


@registry_ignore_patterns('inputs')
def create_input_mul_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Multiply', 'multiply')
    return pattern.set_name('input_multiply').pattern


@registry_ignore_patterns('inputs')
def create_input_transpose_mul_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.append_single_op('Multiply', 'multiply')
    return pattern.set_name('input_transpose_multiply').pattern


@registry_ignore_patterns('inputs')
def create_input_convert_transpose_mul_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Convert', 'convert')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.append_single_op('Multiply', 'multiply')
    return pattern.set_name('input_convert_transpose_multiply').pattern


@registry_ignore_patterns('inputs')
def create_input_reverse_input_channel_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_scaleshift()
    return pattern.set_name('input_reverse_input_channels_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_transpose_reverse_input_channel_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_scaleshift()
    return pattern.set_name('input_transpose_reverse_input_channels_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_convert_transpose_reverse_input_channel_scaleshift_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Convert', 'convert')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_scaleshift()
    return pattern.set_name('input_convert_transpose_reverse_input_channels_scale_shift').pattern


@registry_ignore_patterns('inputs')
def create_input_reverse_input_channel_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_add_const()
    return pattern.set_name('input_reverse_input_channels_add').pattern


@registry_ignore_patterns('inputs')
def create_input_transpose_reverse_input_channel_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_add_const()
    return pattern.set_name('input_transpose_reverse_input_channels_add').pattern


@registry_ignore_patterns('inputs')
def create_input_convert_transpose_reverse_input_channel_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Parameter', 'input')
    pattern.append_single_op('Convert', 'convert')
    pattern.append_single_op('Transpose', 'transpose')
    pattern.insert_split()
    pattern.append_single_op('Concat', 'concat')
    pattern.insert_add_const()
    return pattern.set_name('input_convert_transpose_reverse_input_channels_add').pattern


@registry_ignore_patterns('blocks')
def create_softmax_pattern():
    pattern = PatternBuilder()
    exp_out = pattern.append_single_op('Exp', 'exp').get_last_node()
    pattern.append_op_const('ReduceSum', 'reduce')
    power_out = pattern.append_op_const('Power', 'power').get_last_node()
    pattern.insert_single_op([exp_out, power_out], None, 'Multiply', 'mul')
    return pattern.set_name('softmax').pattern


@registry_ignore_patterns('blocks')
def create_softmax_div_pattern():
    pattern = PatternBuilder()
    exp_out = pattern.append_single_op('Exp', 'exp').get_last_node()
    reduce_out = pattern.append_op_const('ReduceSum', 'reduce').get_last_node()
    pattern.insert_single_op([exp_out, reduce_out], None, 'Divide', 'div')
    return pattern.set_name('softmax_div').pattern


@registry_ignore_patterns('blocks')
def create_softmax_reshape_matmul_pattern():
    pattern = PatternBuilder()
    pattern_2 = PatternBuilder()
    pattern.append_single_op('SoftMax', 'softmax')
    reshape_out = pattern.append_op_const('Reshape', 'softmax').get_last_node()
    pattern_2.append_single_op('Add', 'add').get_last_node()
    pattern_2.append_op_const('Reshape', 'reshape')
    transp_out = pattern_2.append_single_op('Transpose', 'transpose').get_last_node()
    pattern.pattern['nodes'] += pattern_2.pattern['nodes']
    pattern.pattern['edges'] += pattern_2.pattern['edges']
    pattern.insert_single_op([transp_out, reshape_out], None, 'MatMul', 'matmul')
    return pattern.set_name('softmax_reshape_matmul').pattern


# Stable diffusion UNet
@registry_ignore_patterns('blocks')
def create_stable_diffusion_pattern():
    pattern = PatternBuilder()
    pattern_2 = PatternBuilder()
    softmax_out = pattern.append_single_op('SoftMax', 'softmax').get_last_node()
    pattern_2.append_single_op('Reshape', 'reshape1')
    pattern_2.append_single_op('Transpose', 'transpose')
    transp_out = pattern_2.append_single_op('Reshape', 'reshape2').get_last_node()
    pattern.pattern['nodes'] += pattern_2.pattern['nodes']
    pattern.pattern['edges'] += pattern_2.pattern['edges']
    pattern.insert_single_op([transp_out, softmax_out], None, 'MatMul', 'matmul')
    return pattern.set_name('stable_diffusion').pattern


@registry_ignore_patterns('blocks')
def create_hswish_without_denominator_pattern():
    pattern = PatternBuilder()
    pattern.insert_conv_fc(name='input')
    hswish_input = pattern.insert_bias().get_last_node()
    pattern.insert_add_const()
    clamp_out = pattern.append_single_op('Clamp', 'clamp').get_last_node()
    pattern.insert_single_op([hswish_input, clamp_out], None, 'Multiply', 'mul')
    return pattern.set_name('hswish_activation_without_denominator').pattern


@registry_ignore_patterns('blocks')
def create_hswish_pattern():
    pattern = PatternBuilder()
    pattern.insert_conv_fc(name='input')
    hswish_input = pattern.insert_bias().get_last_node()
    pattern.insert_add_const()
    clamp_out = pattern.append_single_op('Clamp', 'clamp').get_last_node()
    pattern.insert_single_op([hswish_input, clamp_out], None, 'Multiply', 'mul')
    pattern.insert_multiply_const()
    return pattern.set_name('hswish_activation').pattern


@registry_ignore_patterns('blocks')
def create_hswish_pattern_2():
    pattern = PatternBuilder()
    op_types = lambda x: x in ['Add', 'Multiply', 'ReduceMean', 'Squeeze']
    hswish_input = pattern.insert_single_op(None, None, op_types, op_name='input').get_last_node()
    pattern.insert_add_const(input_node=hswish_input)
    pattern.append_single_op('Clamp', 'clamp')
    clamp_out_mult_const = pattern.insert_multiply_const().get_last_node()
    pattern.insert_single_op([hswish_input, clamp_out_mult_const], None, 'Multiply', 'mul')
    return pattern.set_name('hswish_activation_v2').pattern


@registry_ignore_patterns('blocks')
def create_fc_bn_hswish_pattern():
    pattern = PatternBuilder()
    pattern.insert_op_const(input_node=None, output_node=None, op_type='Unsqueeze', op_name='unsqueeze')
    pattern.insert_multiply_const()
    pattern.insert_add_const()
    pattern.insert_op_const(None, None, op_type='Squeeze', op_name='squeeze')
    return pattern.set_name('fc_bn_hswish_activation').pattern


@registry_ignore_patterns('blocks')
def create_batch_index_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Subtract', 'subtract')
    pattern.append_single_op('Multiply', 'multiply')
    pattern.insert_multiply_const()
    pattern.append_single_op('Add', 'add')
    pattern.append_single_op('Unsqueeze', 'unsqueeze')
    pattern.append_single_op('Concat', 'concat')
    # the second concat has constant input
    # this is the new dimension with batch index
    pattern.append_op_const('Concat')
    pattern.append_single_op('Reshape', 'reshape')
    pattern.append_single_op('Convolution', 'convolution')
    return pattern.set_name('batch_index').pattern


@registry_ignore_patterns('blocks')
def create_experimentaldetectrondetectionoutput_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('ExperimentalDetectronDetectionOutput', 'ExperimentalDetectronDetectionOutput')
    pattern.insert_add_const()
    return pattern.set_name('experimental_detectron_detection_output_add').pattern


@registry_ignore_patterns('blocks')
def create_experimentaldetectronroifeatureextractor_add_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('ExperimentalDetectronROIFeatureExtractor', 'ExperimentalDetectronROIFeatureExtractor')
    pattern.insert_add_const()
    return pattern.set_name('experimental_detectron_roi_feature_extractor_add').pattern

@registry_ignore_patterns('blocks')
def create_equal_logicalnot_pattern():
    pattern = PatternBuilder()
    pattern.append_single_op('Equal', 'equal')
    pattern.append_single_op('LogicalNot', 'logicalnot')
    return pattern.set_name('equal_logicalnot').pattern
