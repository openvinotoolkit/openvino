# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.pot.graph.pattern_utils import get_assign_result_pattern


def get_gna_ignored_patterns():
    return {
        'blocks': [get_assign_result_pattern()],
        'activations': [],
        'inputs': []
    }
