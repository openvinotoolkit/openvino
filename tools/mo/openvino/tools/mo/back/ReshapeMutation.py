# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.ForceStrictPrecision import ForceStrictPrecision
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class ReshapeMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape = match['reshape']

        if reshape.soft_get('type') == 'Reshape':
            reshape['force_precision_in_ports'] = {1: 'int64'}
