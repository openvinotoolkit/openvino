# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.graph.pattern_utils import get_assign_result_pattern, \
    get_softmax_reshape_transpose_gather_matmul_pattern


def get_gna_ignored_patterns():
    return {
        'blocks': [get_assign_result_pattern(), get_softmax_reshape_transpose_gather_matmul_pattern()],
        'activations': [],
        'inputs': []
    }


def get_gna3_ignored_patterns():
    return {
        'blocks': [get_assign_result_pattern(), get_softmax_reshape_transpose_gather_matmul_pattern()],
        'activations': [],
        'inputs': []
    }
