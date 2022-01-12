# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
from openvino.frontend import ModelAnalysisData        # pylint: disable=no-name-in-module,import-error
from openvino.runtime import PartialShape        # pylint: disable=no-name-in-module,import-error
from openvino.runtime.utils.types import get_dtype
#from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_precision

def json_model_analysis_dump(analysis_data: ModelAnalysisData):

    def dump_partial_shape(shape: PartialShape):
        return [dim.get_length() if dim.is_static else '<UNKNOWN>' for dim in shape]

    json_dump = {}
    json_dump['inputs'] = {}
    for (input_name, shape), (_, type) in zip(analysis_data.inputs_shape_map.items(), analysis_data.inputs_type_map.items()):
        json_dump['inputs'][input_name] = {}
        json_dump['inputs'][input_name]['shape'] = dump_partial_shape(shape)
        json_dump['inputs'][input_name]['data_type'] = str(get_dtype(type))
        json_dump['inputs'][input_name]['value'] = 'None' # not supported in 22.1

    json_model_analysis_print(json_dump)
    print(json_dump)


def json_model_analysis_print(json_dump:str):
    print(json_dump)
