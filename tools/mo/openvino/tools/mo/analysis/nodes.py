# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.model_analysis import AnalyzeAction


class IntermediatesNodesAnalysis(AnalyzeAction):
    """
    The analyser gets node names, their shapes and values (if possible) of all nodes in the model.
    """
    def analyze(self, graph: Graph):
        outputs_desc = dict()

        for node in graph.get_op_nodes():
            outputs_desc[node.name] = {'shape': node.soft_get('shape', None),
                                     'data_type': None,
                                     'value': None,
                                     }
        return {'intermediate': outputs_desc}, None
