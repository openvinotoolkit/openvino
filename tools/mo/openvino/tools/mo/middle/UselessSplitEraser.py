# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class UselessSplitEraser(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('split', {'op': 'Split', 'num_splits': 1})],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['split']
        name = node.soft_get('name', node.id)

        assert node.soft_get('input_port', 0) == 0, \
            'Internal attribute `input_port` was not resolved on front phase, broken Split {}'.format(name)
        assert len(node.out_ports()) == 1

        node.out_port(0).get_connection().set_source(node.in_port(0).get_connection().get_source())
