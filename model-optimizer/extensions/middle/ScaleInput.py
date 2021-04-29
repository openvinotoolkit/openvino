# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class ScaleInput(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
        return [AddMeanScaleValues]

    def pattern(self):
        return dict(
            nodes=[
                ('placeholder', dict(kind='op', op='Parameter')),
                ('data', dict(kind='data'))],
            edges=[
                ('placeholder', 'data'),
            ],
        )

    def replace_pattern(self, graph: Graph, match: dict):
        scale = graph.graph['cmd_params'].scale
        if scale is None or scale == 1:
            return
        assert (len(match['placeholder'].out_nodes()))

        AddMeanScaleValues.apply_scale(graph, match['placeholder'], {'scale': np.array([scale])})
