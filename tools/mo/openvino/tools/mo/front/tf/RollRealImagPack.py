# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import add_constant_to_negative_values
from openvino.tools.mo.graph.graph import Graph


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

    def run_before(self):
        from openvino.tools.mo.front.Pack import Pack
        return [Pack]

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
        add_constant_to_negative_values(unroll, 2, int64_array(-1))
        pack = match['pack']
        pack.out_port(0).get_connection().set_source(unroll.out_port(0))
        graph.remove_nodes_from([match['real'].id, match['imag'].id])
