# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.model_analysis import AnalyzeAction


class TrainingPhaseAnalysis(AnalyzeAction):

    def analyze(self, graph: Graph):
        nodes = graph.get_op_nodes(op='Parameter', data_type=bool)
        names = ""
        params = ""
        if not nodes:
            return None, None

        for node in nodes:
            names = names + '\t{}\n'.format(node.name)
            params = params + '\t--input "{}->False" or --input "{}->True"\n'.format(node.name,
                                                                                     node.name)

        message = 'It looks like there are input nodes of boolean type:\n' + names

        message = message + 'If this input node is as switch between the training and an inference mode, ' \
                            'then you need to freeze this input with value True or False.\n' \
                            'In order to do this run the Model Optimizer with the command line parameter:\n' \
                  + params

        message = message + 'to switch graph to inference mode.'
        return None, message
