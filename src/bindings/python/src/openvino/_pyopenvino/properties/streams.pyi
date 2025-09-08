# type: ignore
from __future__ import annotations
import openvino._pyopenvino
import typing
"""
openvino.properties.streams submodule that simulates ov::streams
"""
__all__ = ['Num', 'num']
class Num:
    AUTO: typing.ClassVar[Num]  # value = <openvino._pyopenvino.properties.streams.Num object>
    NUMA: typing.ClassVar[Num]  # value = <openvino._pyopenvino.properties.streams.Num object>
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: typing.SupportsInt) -> None:
        ...
    def to_integer(self) -> int:
        ...
@typing.overload
def num() -> str:
    ...
@typing.overload
def num(arg0: Num) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def num(arg0: typing.SupportsInt) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
