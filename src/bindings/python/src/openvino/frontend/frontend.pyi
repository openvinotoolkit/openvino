# type: ignore
from __future__ import annotations
from openvino._ov_api import Model
from openvino._pyopenvino import FrontEnd as FrontEndBase
from openvino._pyopenvino import FrontEndManager as FrontEndManagerBase
from openvino._pyopenvino import InputModel
import openvino._ov_api
import openvino._pyopenvino
__all__ = ['FrontEnd', 'FrontEndBase', 'FrontEndManager', 'FrontEndManagerBase', 'InputModel', 'Model']
class FrontEnd(openvino._pyopenvino.FrontEnd):
    def __init__(self, fe: openvino._pyopenvino.FrontEnd) -> None:
        ...
    def convert(self, model: typing.Union[openvino._ov_api.Model, openvino._pyopenvino.InputModel]) -> openvino._ov_api.Model:
        ...
    def convert_partially(self, model: openvino._pyopenvino.InputModel) -> openvino._ov_api.Model:
        ...
    def decode(self, model: openvino._pyopenvino.InputModel) -> openvino._ov_api.Model:
        ...
    def normalize(self, model: openvino._ov_api.Model) -> None:
        ...
class FrontEndManager(openvino._pyopenvino.FrontEndManager):
    def load_by_framework(self, framework: str) -> typing.Optional[openvino.frontend.frontend.FrontEnd]:
        ...
    def load_by_model(self, model: str) -> typing.Optional[openvino.frontend.frontend.FrontEnd]:
        ...
