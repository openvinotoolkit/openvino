"""
 Copyright (c) 2019 Intel Corporation

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
from mo.graph.graph import Graph
from mo.utils.model_analysis import AnalyzeAction


class IntermediatesNodesAnalysis(AnalyzeAction):
    """
    The analyser gets node names, their shapes and values (if possible) of all nodes in the model.
    """
    def analyze(self, graph: Graph):
        outputs_desc = dict()

        for node in graph.get_op_nodes():
            outputs_desc[node.id] = {'shape': node.soft_get('shape', None),
                                     'data_type': None,
                                     'value': None,
                                     }
        return {'intermediate': outputs_desc}
