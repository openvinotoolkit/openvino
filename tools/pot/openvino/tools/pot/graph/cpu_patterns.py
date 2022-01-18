# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pattern_utils import check_fused_scale_shift_patterns, get_fused_scale_shift_patterns, \
    check_fused_op_const_patterns, get_fused_op_const_pattern, get_clamp_mult_const_pattern


def get_cpu_ignored_patterns():
    return {
        'blocks': [(pattern, check_fused_scale_shift_patterns) for pattern in get_fused_scale_shift_patterns()] +
                  [(pattern, check_fused_op_const_patterns) for pattern in get_fused_op_const_pattern()],
        'activations': [get_clamp_mult_const_pattern()],
        'inputs': []
    }
