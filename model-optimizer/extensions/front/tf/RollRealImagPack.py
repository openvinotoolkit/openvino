# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from extensions.middle.InsertLayoutPropagationTransposes import mark_input_as_in_correct_layout
from extensions.ops.roll import Roll
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, Node, rename_nodes


class RollRealImagPack(FrontReplacementSubgraph):
    """
    Some TF models contain Roll for complex data, as a part of the sub-graph

        input   shift   axes
          |       |      |
         -------------------
                 Roll
                  |
         -------------------
         |                 |
        Real              Imag
         |                 |
         -------     -------
               |    |
                Pack
                  |
                SomeOp

    This sub-graph can be replaced with the sub-graph

        input   shift   axes
          |       |      |
         -------------------
                 Roll
                  |
                SomeOp

    But after such replacement, we should correct axes of Roll, because input data are real now. Namely, if
    there are negative axes for Roll, we need subtract 1 from such axes indices.
    """
    enabled = True

    def run_after(self):
        from extensions.front.tf.SSliceComplexRoll import SSliceComplexRoll
        return [SSliceComplexRoll]

    def pattern(self):
        return dict(
            nodes=[
                ('unroll', dict(op='Roll')),
                ('real', dict(op='Real')),
                ('imag', dict(op='Imag')),
                ('pack', dict(op='Pack')),
            ],
            edges=[
                ('unroll', 'real', {'in': 0}),
                ('unroll', 'imag', {'in': 0}),
                ('real', 'pack', {'in': 0}),
                ('imag', 'pack', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        unroll = match['unroll']
        unroll_name = unroll.soft_get('name', unroll.id)

        new_unroll = Roll(graph, {}).create_node()
        self.correct_roll_axes(unroll)

        unroll.in_port(0).get_connection().set_destination(new_unroll.in_port(0))
        unroll.in_port(1).get_connection().set_destination(new_unroll.in_port(1))
        unroll.in_port(2).get_connection().set_destination(new_unroll.in_port(2))

        pack = match['pack']
        pack.out_port(0).get_connection().set_source(new_unroll.out_port(0))

        rename_nodes([(unroll, unroll_name + '/to_be_removed'), (new_unroll, unroll_name)])

    @staticmethod
    def correct_roll_axes(roll: Node):
        axes_node = roll.in_port(2).get_source().node
        if axes_node.soft_get('type') != 'Const':
            return
        axes = axes_node.soft_get('value', None)
        if axes is None:
            return

        corrected_axes = axes.copy()
        for i, axis in enumerate(axes):
            if axis < 0:
                corrected_axes[i] = axis - 1

        axes_node.value = int64_array(corrected_axes)
        mark_input_as_in_correct_layout(roll, 2)
