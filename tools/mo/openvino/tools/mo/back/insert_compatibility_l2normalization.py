# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class CompatibilityL2NormalizationPattern(BackReplacementPattern):
    force_clean_up = True
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('l2_normalization', dict(op='Normalize'))
            ],
            edges=[])

    def replace_pattern(self, graph: Graph, match: dict):
        """
        Adds Normalize layer weights, which are required by OpenVINO, 
        but do not always exist in MXNet model. 
        
        L2Normalization is mapped to Normalize layer
        so we need to generate Normalize weights filled with ones.
        
        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        l2_normalization_node = match['l2_normalization']
        if len(l2_normalization_node.in_nodes()) < 2:
            value = np.full([l2_normalization_node.in_node(0).shape[1]], 1.0, dtype=np.float32)
            weights_node = Const(graph, dict(name=l2_normalization_node['name'] + '_weights', value=value)).create_node()
            l2_normalization_node.add_input_port(1)
            l2_normalization_node.in_port(1).connect(weights_node.out_port(0))
            l2_normalization_node.in_port(1).bin = 'weights'
