# type: ignore
from __future__ import annotations
import openvino._pyopenvino
import typing
"""
openvino.properties.intel_auto submodule that simulates ov::intel_auto
"""
__all__ = ['SchedulePolicy', 'device_bind_buffer', 'enable_runtime_fallback', 'enable_startup_fallback', 'schedule_policy']
class SchedulePolicy:
    """
    Members:
    
      ROUND_ROBIN
    
      DEVICE_PRIORITY
    
      DEFAULT
    """
    DEFAULT: typing.ClassVar[SchedulePolicy]  # value = <SchedulePolicy.DEVICE_PRIORITY: 1>
    DEVICE_PRIORITY: typing.ClassVar[SchedulePolicy]  # value = <SchedulePolicy.DEVICE_PRIORITY: 1>
    ROUND_ROBIN: typing.ClassVar[SchedulePolicy]  # value = <SchedulePolicy.ROUND_ROBIN: 0>
    __members__: typing.ClassVar[dict[str, SchedulePolicy]]  # value = {'ROUND_ROBIN': <SchedulePolicy.ROUND_ROBIN: 0>, 'DEVICE_PRIORITY': <SchedulePolicy.DEVICE_PRIORITY: 1>, 'DEFAULT': <SchedulePolicy.DEVICE_PRIORITY: 1>}
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
def device_bind_buffer() -> str:
    ...
@typing.overload
def device_bind_buffer(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def enable_runtime_fallback() -> str:
    ...
@typing.overload
def enable_runtime_fallback(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def enable_startup_fallback() -> str:
    ...
@typing.overload
def enable_startup_fallback(arg0: bool) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
@typing.overload
def schedule_policy() -> str:
    ...
@typing.overload
def schedule_policy(arg0: SchedulePolicy) -> tuple[str, openvino._pyopenvino.OVAny]:
    ...
