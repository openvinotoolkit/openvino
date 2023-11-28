# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.graph.pattern_utils import get_clamp_mult_const_pattern, \
    get_softmax_reshape_transpose_gather_matmul_pattern, get_softmax_reshape_transpose_matmul_pattern

def get_npu_ignored_patterns():
    return {
        'blocks': [get_softmax_reshape_transpose_gather_matmul_pattern(), get_softmax_reshape_transpose_matmul_pattern()],
        'activations': [get_clamp_mult_const_pattern()],
        'inputs': []
    }
