# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.mxnet.extractors.utils import get_json_layer_attrs
from openvino.tools.mo.graph.graph import Graph


class SsdPatternRemoveReshape(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('multi_box_prior', dict(op='_contrib_MultiBoxPrior')),
                ('concat', dict(op='Concat')),
                ('reshape', dict(op='Reshape'))
            ],
            edges=[
                ('multi_box_prior', 'concat', {'in': 0}),
                ('concat', 'reshape', {'in': 0})
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        Need to find each occurrence of pattern: _contrib_MultiBoxPrior(s) -> Concat -> Reshape
        remove Reshape layer - OV does not expect outputs from concatenation of _contrib_MultiBoxPrior to be reshaped

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        reshape_node = match['reshape']
        reshape_node.out_port(0).get_connection().set_source(reshape_node.in_port(0).get_connection().get_source())
        graph.remove_node(reshape_node.id)

        # concat should be performed for the third axis
        concat_node = match['concat']
        attr = get_json_layer_attrs(concat_node.graph.node[concat_node.id]['symbol_dict'])
        if 'dim' in attr:
            attr['dim'] = 2
            concat_node['axis'] = 2
