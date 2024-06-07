# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging as log
import sys

import numpy as np

from openvino.tools.mo.front.user_data_repack import UserDataRepack
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_precision
from openvino.tools.mo.utils.model_analysis import AnalyzeAction, AnalysisCollectorAnchor, AnalysisResults


def prepare_obj_for_dump(obj: object):
    if isinstance(obj, dict):
        return {k: prepare_obj_for_dump(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        else:
            return [prepare_obj_for_dump(elem) for elem in obj]
    elif isinstance(obj, list):
        return [prepare_obj_for_dump(elem) for elem in obj]
    elif isinstance(obj, type):
        try:
            return np_data_type_to_precision(obj)
        except:
            log.error('Unsupported data type: {}'.format(str(obj)))
            return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return str(obj)


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
        analysis_results = AnalysisResults()
        if analysis_results.get_result() is not None:
            try:
                print(json.dumps(prepare_obj_for_dump(analysis_results.get_result())))
            except Exception as e:
                log.error('Cannot serialize to JSON: %s', str(e))
                sys.exit(1)
        sys.exit(0)

