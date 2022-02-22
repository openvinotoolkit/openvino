# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.graph.pattern_utils import get_clamp_mult_const_pattern

def get_vpu_ignored_patterns():
    return {
        'blocks': [],
        'activations': [get_clamp_mult_const_pattern()],
        'inputs': []
    }
