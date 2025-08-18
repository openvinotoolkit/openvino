# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Op as OpBase
from openvino._pyopenvino import Output
import openvino._pyopenvino
__all__ = ['Node', 'Op', 'OpBase', 'Output']
class Op(openvino._pyopenvino.Op):
    def __init__(self, py_obj: Op, inputs: typing.Union[list[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]], tuple[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output, list[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]]]], NoneType] = None) -> None:
        ...
