# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.FakeQuantWithMinMaxVars import FakeQuantWithMinMaxVarsToQuantize
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class DisableQuantizeValuePropagation(FrontReplacementPattern):
    enabled = True

    def run_after(self):
        return [FakeQuantWithMinMaxVarsToQuantize]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('quantize', dict(op='FakeQuantize', levels=lambda levels: levels != 2)),
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        match['quantize']['stop_value_propagation'] = True
