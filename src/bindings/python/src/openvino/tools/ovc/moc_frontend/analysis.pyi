from __future__ import annotations
import json as json
import openvino._ov_api
from openvino._ov_api import Model
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Type
from openvino.utils.types import get_dtype
__all__ = ['Model', 'PartialShape', 'Type', 'get_dtype', 'json', 'json_model_analysis_dump', 'json_model_analysis_print']
def json_model_analysis_dump(framework_model: openvino._ov_api.Model):
    ...
def json_model_analysis_print(json_dump: str):
    ...
