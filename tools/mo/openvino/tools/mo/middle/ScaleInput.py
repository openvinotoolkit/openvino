# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.AddMeanScaleValues import AddMeanScaleValues
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ScaleInput(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.AddMeanScaleValues import AddMeanScaleValues
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

        AddMeanScaleValues.apply_scale(graph, match['placeholder'], {'scale': mo_array([scale])})
