# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Op as OpBase
from openvino._pyopenvino import Output
from typing import Any
import openvino._pyopenvino
import typing
__all__ = ['Any', 'Node', 'Op', 'OpBase', 'Output']
class Op(openvino._pyopenvino.Op):
    def __init__(self, py_obj: Op, inputs: typing.Any = None) -> None:
        ...
