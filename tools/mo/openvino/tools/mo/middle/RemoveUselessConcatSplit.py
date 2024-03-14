# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class RemoveUselessConcatSplitPattern(MiddleReplacementPattern):
    r"""
    Remove useless construction with concat and split like follows:
         /    /   |    \     \
       br1  br2   ..  br(n-1)br(n)
        \    \    |    /    /
                concat
                  |
                split
         /    /   |    \     \
       br1  br2   ..  br(n-1)br(n)

    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.ReplaceSpliceNodePattern import ReplaceSpliceNodePattern
        return [ReplaceSpliceNodePattern]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('concat', dict(op='Concat')),
                   ('data', dict(kind='data')),
                   ('split', dict(op='Split'))],
            edges=[('concat', 'data'),
                   ('data', 'split')])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        concat_node = match['concat']
        split_node = match['split']

        # don't apply pass if concat have another outputs except split
        if len(concat_node.out_port(0).get_destinations()) != 1:
            return

        inputs = list(concat_node.in_ports().values())
        outputs = list(split_node.out_ports().values())

        if len(inputs) != len(outputs):
            return

        for i in range(len(inputs)):
            if not all(inputs[i].data.get_shape() == outputs[i].data.get_shape()):
                return

        for i in range(len(inputs)):
            outputs[i].get_connection().set_source(inputs[i].get_source())
            inputs[i].disconnect()
