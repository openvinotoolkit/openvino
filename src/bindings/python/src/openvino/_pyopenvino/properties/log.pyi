# type: ignore
from __future__ import annotations
import openvino._pyopenvino
import typing
"""
openvino.properties.log submodule that simulates ov::log
"""
__all__ = ['Level', 'level']
class Level:
    """
    Members:
    
      NO
    
      ERR
    
      WARNING
    
      INFO
    
      DEBUG
    
      TRACE
    """
    DEBUG: typing.ClassVar[Level]  # value = <Level.DEBUG: 3>
    ERR: typing.ClassVar[Level]  # value = <Level.ERR: 0>
    INFO: typing.ClassVar[Level]  # value = <Level.INFO: 2>
    NO: typing.ClassVar[Level]  # value = <Level.NO: -1>
    TRACE: typing.ClassVar[Level]  # value = <Level.TRACE: 4>
    WARNING: typing.ClassVar[Level]  # value = <Level.WARNING: 1>
    __members__: typing.ClassVar[dict[str, Level]]  # value = {'NO': <Level.NO: -1>, 'ERR': <Level.ERR: 0>, 'WARNING': <Level.WARNING: 1>, 'INFO': <Level.INFO: 2>, 'DEBUG': <Level.DEBUG: 3>, 'TRACE': <Level.TRACE: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
@typing.overload
def level() -> str:
    ...
@typing.overload
def level(arg0: Level) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
