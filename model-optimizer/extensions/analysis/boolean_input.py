"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from mo.graph.graph import Graph
from mo.utils.model_analysis import AnalyzeAction


class TrainingPhaseAnalysis(AnalyzeAction):

    def analyze(self, graph: Graph):
        nodes = graph.get_op_nodes(op='Parameter', data_type=np.bool)
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
