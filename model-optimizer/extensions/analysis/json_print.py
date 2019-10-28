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
import json
import sys

import numpy as np

from extensions.front.user_data_repack import UserDataRepack
from mo.graph.graph import Graph
from mo.middle.passes.convert_data_type import np_data_type_to_precision
from mo.utils.model_analysis import AnalyzeAction, AnalysisCollectorAnchor


def prepare_obj_for_dump(obj: object):
    if isinstance(obj, dict):
        return {k: prepare_obj_for_dump(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray) or isinstance(obj, list):
        return [prepare_obj_for_dump(elem) for elem in obj]
    elif isinstance(obj, type):
        return np_data_type_to_precision(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class AnalysisJSONPrint(AnalyzeAction):
    """
    The action prints the analysis results in JSON format.
    """
    enabled = False
    id = 'ANALYSIS_JSON_PRINT'

    def run_before(self):
        return [UserDataRepack]

    def run_after(self):
        return [AnalysisCollectorAnchor]

    def analyze(self, graph: Graph):
        if 'analysis_results' in graph.graph and graph.graph['analysis_results'] is not None:
            print(json.dumps(prepare_obj_for_dump(graph.graph['analysis_results'])))
        sys.exit(0)
