# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Op as OpBase
from openvino._pyopenvino import Output
import openvino._pyopenvino
__all__ = ['Node', 'Op', 'OpBase', 'Output']
class Op(openvino._pyopenvino.Op):
    def __init__(self, py_obj: Op, inputs: typing.Union[typing.List[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]], typing.Tuple[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output, typing.List[typing.Union[openvino._pyopenvino.Node, openvino._pyopenvino.Output]]]], NoneType] = None) -> None:
        ...
